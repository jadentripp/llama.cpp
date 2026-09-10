#!/usr/bin/env python3
"""Compare resident and compact streaming logits using a converted Edge0 bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("bundle", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prompt-file", type=Path, required=True)
parser.add_argument("--gpu-layers", type=int, default=0)
parser.add_argument("--tokens", type=int, default=8)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
outputs = []
for mode in ("resident", "stream"):
    path = args.output / (mode + ".f32")
    env = dict(os.environ, GGML_MOE_STREAM="1" if mode == "stream" else "0",
               GGML_MOE_STREAM_CPU="1" if args.gpu_layers == 0 else "0",
               GGML_MOE_CACHE_MIB="16", GGML_MOE_TRACE="1")
    command = [str(args.executable.resolve()), "--bundle", str(args.bundle.resolve()),
               "--prompt-file", str(args.prompt_file.resolve()), "--tokens", str(args.tokens),
               "--gpu-layers", str(args.gpu_layers), "--context", "512", "--batch", "32",
               "--verify-replay", "--logits-file", str(path.resolve())]
    with (args.output / (mode + ".txt")).open("w") as out, (args.output / (mode + ".log")).open("w") as log:
        subprocess.run(command, env=env, stdout=out, stderr=log, check=True)
    outputs.append(np.fromfile(path, dtype=np.float32))
resident, stream = outputs
if not resident.size or resident.shape != stream.shape or not np.isfinite(resident).all() or not np.isfinite(stream).all():
    raise RuntimeError("missing, differently sized, or non-finite logits")
delta = resident.astype(np.float64) - stream
nmse = float(np.dot(delta, delta) / max(np.dot(resident.astype(np.float64), resident), 1e-12))
result = dict(values=int(resident.size), max_abs_error=float(np.abs(delta).max()), normalized_squared_error=nmse)
(args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if nmse > 5e-5:
    raise RuntimeError("streaming/resident logits differ beyond the backend tolerance")
