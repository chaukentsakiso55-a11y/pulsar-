import torch

from model.pulsar1 import PulsarConfig, PulsarLM


def test_model_forward_shape():
    cfg = PulsarConfig(vocab_size=259, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = PulsarLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
