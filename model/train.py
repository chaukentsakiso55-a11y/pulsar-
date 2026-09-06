from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from model.pulsar1 import PulsarConfig, PulsarLM
from model.tokenizer import ByteTokenizer


def load_config(path: str) -> PulsarConfig:
    return PulsarConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pulsar-1 on UTF-8 text")
    parser.add_argument("--config", default="configs/pulsar-1-tiny.json")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--out", default="checkpoints/pulsar-1.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    tokenizer = ByteTokenizer()
    text = Path(args.data).read_text(encoding="utf-8")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(tokens) <= cfg.block_size + 1:
        raise SystemExit(
            f"Training text is too short. Need > {cfg.block_size + 1} byte tokens; got {len(tokens)}."
        )

    model = PulsarLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} parameters={model.parameter_count():,}")

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(tokens) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([tokens[i : i + cfg.block_size] for i in starts]).to(device)
        y = torch.stack([tokens[i + 1 : i + cfg.block_size + 1] for i in starts]).to(device)
        return x, y

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch()
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step={step} loss={loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "training": {
                "steps": args.steps,
                "data": str(args.data),
                "seed": args.seed,
            },
        },
        out,
    )
    print(f"saved={out}")


if __name__ == "__main__":
    main()
