from __future__ import annotations

import argparse

import torch

from model.pulsar1 import PulsarConfig, PulsarLM
from model.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a Pulsar-1 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="<user>\nHello Pulsar\n<assistant>\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = PulsarConfig(**payload["config"])
    model = PulsarLM(cfg).to(device)
    model.load_state_dict(payload["model"])
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(args.prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
