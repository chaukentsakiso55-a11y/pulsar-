from __future__ import annotations


class ByteTokenizer:
    """Zero-dependency byte tokenizer used by Pulsar-1 v0.1.

    IDs 0-255 map to raw UTF-8 bytes. 256-258 are special tokens.
    """

    PAD = 256
    BOS = 257
    EOS = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS)
        ids.extend(text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(i for i in ids if 0 <= i <= 255)
        return data.decode("utf-8", errors="replace")
