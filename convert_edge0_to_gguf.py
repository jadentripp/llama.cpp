#!/usr/bin/env python3
"""Convert a complete, matched Edge0 release to GGUF base/LoRA/prerouter files.

Requires the regular HF converter dependencies. Uses the upstream metadata and
tokenizer conversion, but preserves MLX affine int4 coefficients and codes.
See docs/backend/EDGE0-PORT.md for supported scope and validation limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
import gguf
from conversion.edge0 import SafeTensors, LossyCoefficients, affine_dequant, affine_q4_1, qwen_layout

REFERENCE = "a38d3dae7ed9c24d44f62c455f3c7ac67345d844"


class Converter:
    def __init__(self, directory: Path, output: Path):
        from conversion.qwen import Qwen3_5MoeTextModel
        from conversion.bailingmoe3 import BailingMoeV3Model
        self.directory, self.output = directory, output
        config = json.loads((directory / "config.json").read_text())
        arch = config["architectures"][0]
        if arch == "Qwen3_5MoeForConditionalGeneration":
            self.family, cls = "qwen35", Qwen3_5MoeTextModel
            self.hp = config["text_config"]
            self.owners, self.top_k, suffix = list(range(6, 39)), 4, "35b"
            expected = (40, 2048, 256)
        elif arch == "BailingMoeV3ForCausalLM":
            self.family, cls = "ling3", BailingMoeV3Model
            self.hp = config
            self.owners, self.top_k, suffix = list(range(7, 23)), 8, "8b"
            expected = (24, 1536, 128)
        else:
            raise ValueError(f"unsupported Edge0 architecture: {arch}")
        actual = tuple(self.hp[k] for k in ("num_hidden_layers", "hidden_size", "num_experts"))
        if actual != expected:
            raise ValueError(f"checkpoint dimensions {actual} do not match released {suffix} profile {expected}")
        for field in ("quantization", "quantization_config"):
            q = config.get(field, self.hp.get(field, {}))
            if q and (q.get("group_size", 64) != 64 or q.get("mode", "affine") != "affine"):
                raise ValueError("expected MLX affine group-64 checkpoint")
        self.lora_path = directory / f"lora_edge0_{suffix}.safetensors"
        self.head_path = directory / f"prerouter_edge0_{suffix}.safetensors"
        paths = sorted(directory.glob("model*.safetensors"))
        if not paths or not self.lora_path.is_file() or not self.head_path.is_file():
            raise ValueError("a complete release needs model shards, Recover-LoRA, and prerouter safetensors")
        self.base = SafeTensors(paths)
        self.lora = SafeTensors([self.lora_path])
        self.heads = SafeTensors([self.head_path])

        # Hash actual files, not just dimensions or tensor names. All three GGUFs
        # carry this identifier; the experimental driver checks the complete set.
        identity = hashlib.sha256((REFERENCE + json.dumps(config, sort_keys=True)).encode())
        for path in [*paths, self.lora_path, self.head_path]:
            logging.info("Hashing %s", path.name)
            with path.open("rb") as f:
                identity.update(hashlib.file_digest(f, "sha256").digest())
        self.bundle = identity.hexdigest()

        class MetadataOnly(cls):
            model_arch = cls.model_arch
            no_mtp = True
            def index_tensors(self, remote_hf_model_id=None):
                return {}

            def get_vocab_base(self):
                # The Qwen release names the Transformers 5 TokenizersBackend
                # class. Its tokenizer.json also works with this fork's pinned
                # Transformers 4 fast tokenizer, without remote model code.
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast.from_pretrained(self.dir_model)
                return self._get_vocab_base(tokenizer)

        self.model = MetadataOnly(directory, gguf.LlamaFileType.MOSTLY_Q4_1, output / "base.gguf", hparams=config)
        self.model.hparams["num_experts_per_tok"] = self.top_k
        self.mapping = self.model.tensor_map

    def canonical(self, name):
        name = name.removeprefix("language_model.").replace(".switch_mlp.", ".experts.")
        name = name.replace("_conv1d.conv.weight", "_conv1d.weight")
        if name.endswith(".expert_bias"):
            name += ".bias"
        return name

    def mapped(self, name):
        if name.endswith(".dt_bias"):
            name = name.removesuffix(".dt_bias") + ".dt_proj.bias"
        bid = re.search(r"\.layers\.(\d+)\.", name)
        if self.family == "ling3" and bid:
            layer = int(bid[1])
            if name.endswith(".attention.f_proj.weight"):
                return f"blk.{layer}.ssm_f_a.weight"
            if name.endswith(".attention.g_proj.weight"):
                return f"blk.{layer}." + ("attn_gate.weight" if self.model.is_full_attention(layer) else "ssm_g_a.weight")
        result = self.mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if result is None:
            raise ValueError(f"unmapped tensor: {name}")
        return result

    def base_tensors(self):
        for original in sorted(self.base.items):
            if original.endswith((".scales", ".biases")) or ".prerouter." in original or ".pregate." in original:
                continue
            name = self.canonical(original)
            data = self.base.get(original)
            packed = False
            if self.base.items[original][0] == "U32":
                prefix = original.removesuffix(".weight")
                scales = self.base.get(prefix + ".scales")
                biases = self.base.get(prefix + ".biases")
                width = scales.shape[-1] * 64
                bits = data.shape[-1] * 32 // width
                if width * bits != data.shape[-1] * 32 or bits not in (4, 8):
                    raise ValueError(f"invalid affine layout: {original}")
                # Transposing MLA's K projection changes the quantization axis.
                # Preserve those values in F32; never requantize them implicitly.
                if bits == 4 and not name.endswith(".attention.kv_b_proj.weight"):
                    try:
                        data = affine_q4_1(data, scales, biases)
                        packed = True
                    except LossyCoefficients:
                        logging.warning("%s: preserving non-fp16 affine coefficients in F32", original)
                        data = affine_dequant(data, scales, biases, bits)
                else:
                    data = affine_dequant(data, scales, biases, bits)
            else:
                data = data.astype(np.float32)
            if self.family == "qwen35":
                if ".conv1d." in name:
                    if data.ndim != 3 or data.shape[-1] != 1:
                        raise ValueError("Qwen checkpoint must use sanitized MLX convolution/norm layout")
                    data = data.squeeze(-1)
                data = qwen_layout(name, data, self.hp, packed=packed)
                if name.endswith(".A_log"):
                    data = -np.exp(data)
                # Published MLX norms already contain the HF +1 adjustment.
            else:
                if "_conv1d.weight" in name:
                    data = data.reshape(1, data.shape[0], 1, self.hp["short_conv_kernel_size"])
                if name.endswith(".A_log"):
                    data = np.exp(data).reshape(-1, 1)
                if name.endswith(".attention.kv_b_proj.weight"):
                    layer = int(re.search(r"\.layers\.(\d+)\.", name)[1])
                    nk, nv = self.hp["qk_nope_head_dim"], self.hp["v_head_dim"]
                    data = data.reshape(self.hp["num_attention_heads"], nk + nv, -1)
                    yield f"blk.{layer}.attn_k_b.weight", np.ascontiguousarray(data[:, :nk].swapaxes(1, 2)), gguf.GGMLQuantizationType.F32
                    yield f"blk.{layer}.attn_v_b.weight", np.ascontiguousarray(data[:, nk:]), gguf.GGMLQuantizationType.F32
                    continue
            qtype = gguf.GGMLQuantizationType.Q4_1 if packed else gguf.GGMLQuantizationType.F32
            yield self.mapped(name), np.ascontiguousarray(data), qtype

    def lora_tensors(self):
        for original in sorted(self.lora.items):
            if not original.endswith(".lora_A"):
                if not original.endswith(".lora_B"):
                    raise ValueError(f"unexpected LoRA tensor: {original}")
                continue
            name = self.canonical(original.removesuffix(".lora_A") + ".weight")
            a = self.lora.get(original).astype(np.float16)
            b = self.lora.get(original.removesuffix("A") + "B").astype(np.float16)
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 16 or b.shape[1] != 16:
                raise ValueError(f"expected release LoRA rank 16: {original}")
            if self.family == "qwen35":
                a = qwen_layout(name, a, self.hp, lora_part="A")
                b = qwen_layout(name, b, self.hp, lora_part="B")
            if name.endswith(".attention.kv_b_proj.weight"):
                layer = int(re.search(r"\.layers\.(\d+)\.", name)[1])
                nh, nk, nv = self.hp["num_attention_heads"], self.hp["qk_nope_head_dim"], self.hp["v_head_dim"]
                b = b.reshape(nh, nk + nv, 16)
                # K is stored transposed for MLA absorption, so exchange A/B.
                ka = b[:, :nk].swapaxes(1, 2)
                kb = np.broadcast_to(a.T, (nh, a.shape[1], 16))
                va = np.broadcast_to(a, (nh, *a.shape))
                vb = b[:, nk:]
                for part, data in (("attn_k_b.weight.lora_a", ka), ("attn_k_b.weight.lora_b", kb),
                                   ("attn_v_b.weight.lora_a", va), ("attn_v_b.weight.lora_b", vb)):
                    yield f"blk.{layer}.{part}", np.ascontiguousarray(data), gguf.GGMLQuantizationType.F16
                continue
            mapped = self.mapped(name)
            if mapped == "token_embd.weight":
                a = a.T
            yield mapped + ".lora_a", np.ascontiguousarray(a), gguf.GGMLQuantizationType.F16
            yield mapped + ".lora_b", np.ascontiguousarray(b), gguf.GGMLQuantizationType.F16

    def head_tensors(self):
        heads = {}
        for original in self.heads.items:
            match = re.fullmatch(r"layers\.(\d+)\.(?:mlp\.prerouter\.)?(fc1|fc2|linear_init)\.weight", original)
            if not match:
                raise ValueError(f"unexpected prerouter tensor: {original}")
            owner, part = int(match[1]), match[2]
            if owner in self.owners:
                heads[owner, part] = self.heads.get(original).astype(np.float16)
        dim, experts = self.hp["hidden_size"], self.hp["num_experts"]
        shapes = {"fc1": (512, dim + 2 * experts), "fc2": (experts, 512), "linear_init": (experts, dim + 2 * experts)}
        for owner in self.owners:
            for part, shape in shapes.items():
                data = heads.get((owner, part))
                if data is None and part == "linear_init" and self.family == "ling3":
                    data = np.zeros(shape, dtype=np.float16)
                if data is None or data.shape != shape or not np.isfinite(data).all():
                    raise ValueError(f"invalid/missing head {owner}.{part}, expected {shape}")
                yield f"edge0.{owner}.{part}", data, gguf.GGMLQuantizationType.F16

    def write(self, filename, generator, writer, metadata=None):
        target = self.output / filename
        if target.exists():
            raise FileExistsError(target)
        partial = target.with_suffix(".gguf.partial")
        logging.info("Checking tensor layouts for %s", filename)
        for name, data, qtype in generator():
            writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=qtype)
        if metadata:
            metadata()
        writer.add_string("edge0.bundle_id", self.bundle)
        writer.add_string("edge0.family", self.family)
        writer.add_string("edge0.reference", REFERENCE)
        writer.add_uint32("edge0.format_version", 1)
        writer.write_header_to_file(path=partial)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        try:
            for name, data, _ in generator():
                logging.info("Writing %s", name)
                writer.write_tensor_data(data)
            writer.close()
            partial.rename(target)
        except BaseException:
            writer.close()
            partial.unlink(missing_ok=True)
            raise

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        # Validate the complete adapter set before writing the large base file.
        list(self.lora_tensors())
        list(self.head_tensors())
        writer = self.model.gguf_writer
        self.write("base.gguf", self.base_tensors, writer, lambda: self.model.prepare_metadata(vocab_only=False))
        lora = gguf.GGUFWriter(None, gguf.MODEL_ARCH_NAMES[self.model.model_arch])
        lora.add_string("general.type", "adapter")
        lora.add_string("adapter.type", "lora")
        lora.add_float32("adapter.lora.alpha", 32.0)
        self.write("lora.gguf", self.lora_tensors, lora)
        heads = gguf.GGUFWriter(None, "edge0_prerouter")
        heads.add_uint32("edge0.hidden_size", self.hp["hidden_size"])
        heads.add_uint32("edge0.expert_count", self.hp["num_experts"])
        heads.add_uint32("edge0.top_k", self.top_k)
        heads.add_uint32("edge0.layer_count", self.hp["num_hidden_layers"])
        heads.add_array("edge0.owners", self.owners)
        self.write("prerouter.gguf", self.head_tensors, heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="complete downloaded Edge0 release directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    Converter(args.model, args.output_dir).run()


if __name__ == "__main__":
    main()
