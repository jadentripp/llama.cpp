"""Numerical checks for Edge0 quantization and adapter layout conversion."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gguf-py")]
import gguf
from conversion.edge0 import affine_dequant, affine_q4_1, qwen_layout
from convert_edge0_to_gguf import Converter


class TestEdge0Conversion(unittest.TestCase):
    def test_affine_int4_is_lossless(self):
        rng = np.random.default_rng(50)
        weight = rng.integers(0, 2**32, (3, 8, 32), dtype=np.uint32)
        scales = rng.integers(1, 128, (3, 8, 4)).astype(np.float32) / 1024
        biases = rng.integers(-128, 128, (3, 8, 4)).astype(np.float32) / 512
        actual = gguf.dequantize(affine_q4_1(weight, scales, biases), gguf.GGMLQuantizationType.Q4_1)
        expected = affine_dequant(weight, scales, biases, 4)
        np.testing.assert_array_equal(actual, expected)

    def test_affine_int8(self):
        weight = np.array([[0xFF800100] * 16], dtype=np.uint32)
        actual = affine_dequant(weight, np.array([[0.25]], dtype=np.float32), np.array([[-1.0]], dtype=np.float32), 8)
        np.testing.assert_array_equal(actual[0, :4], [-1, -0.75, 31, 62.75])

    def test_reject_lossy_coefficients(self):
        weight = np.zeros((1, 8), dtype=np.uint32)
        for scale in (1e10, 1e-10, float("nan"), 1.0001):
            with self.assertRaises(ValueError):
                affine_q4_1(weight, np.array([[scale]], dtype=np.float32), np.zeros((1, 1), dtype=np.float32))

    def test_quantized_head_column_permutation(self):
        rng = np.random.default_rng(51)
        hp = dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64)
        weight = rng.integers(0, 2**32, (8, 32), dtype=np.uint32)
        scales = np.full((8, 4), 0.03125, dtype=np.float32)
        biases = np.full((8, 4), -0.125, dtype=np.float32)
        name = "model.layers.0.linear_attn.out_proj.weight"
        packed = qwen_layout(name, affine_q4_1(weight, scales, biases), hp, packed=True)
        actual = gguf.dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
        expected = affine_dequant(weight, scales, biases, 4).reshape(8, 2, 2, 64).swapaxes(1, 2).reshape(8, 256)
        np.testing.assert_array_equal(actual, expected)

    def test_qwen_lora_permutation(self):
        rng = np.random.default_rng(52)
        hp = dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32)
        for name, rows, cols in (("in_proj_qkv", 256, 64), ("out_proj", 64, 128), ("in_proj_a", 4, 64)):
            name = "model.layers.0.linear_attn." + name + ".weight"
            a, b = rng.normal(size=(16, cols)), rng.normal(size=(rows, 16))
            actual = qwen_layout(name, b, hp, lora_part="B") @ qwen_layout(name, a, hp, lora_part="A")
            expected = qwen_layout(name, b @ a, hp)
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_split_mla_lora(self):
        rng = np.random.default_rng(53)
        a = rng.normal(size=(16, 64)).astype(np.float16)
        b = rng.normal(size=(128, 16)).astype(np.float16)
        name = "model.layers.3.attention.kv_b_proj"
        tensors = {name + ".lora_A": a, name + ".lora_B": b}
        class Reader:
            items = tensors
            def get(self, key): return tensors[key]
        conv = Converter.__new__(Converter)
        conv.family = "ling3"
        conv.hp = dict(num_attention_heads=2, qk_nope_head_dim=32, v_head_dim=32)
        conv.lora = Reader()
        result = {name: data.astype(np.float32) for name, data, _ in conv.lora_tensors()}
        delta = (b.astype(np.float32) @ a.astype(np.float32)).reshape(2, 64, 64)
        for part, expected in (("k", delta[:, :32].swapaxes(1, 2)), ("v", delta[:, 32:])):
            prefix = f"blk.3.attn_{part}_b.weight.lora_"
            actual = result[prefix + "b"] @ result[prefix + "a"]
            np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
