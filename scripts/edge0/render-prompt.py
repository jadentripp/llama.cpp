#!/usr/bin/env python3
"""Render the release's chat prompt for llama-edge0 --prompt-file."""
import argparse
import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("message")
args = parser.parse_args()
config = json.loads((args.model / "tokenizer_config.json").read_text())
template = args.model / "chat_template.jinja"
source = template.read_text() if template.exists() else config["chat_template"]
env = Environment(undefined=StrictUndefined)
def fail(message):
    raise ValueError(message)
env.globals["raise_exception"] = fail
print(env.from_string(source).render(
    messages=[{"role": "user", "content": args.message}], add_generation_prompt=True,
    enable_thinking=False, thinking=False, tools=None,
    bos_token=config.get("bos_token", ""), eos_token=config.get("eos_token", "")), end="")
