
import hashlib
import json
from collections import Counter
from pathlib import Path


class ByteBPETokenizer:
    """A small, from-scratch UTF-8 BPE tokenizer."""

    PAD_ID = 256
    EOS_ID = 257
    FIRST_MERGE_ID = 258

    def __init__(self, merges=None):
        self.merges = [
            tuple(pair) for pair in (merges or [])
        ]

        # The first 256 IDs represent individual bytes.
        self.id_to_bytes = {
            i: bytes([i]) for i in range(256)
        }

        # Reconstruct all learned merge tokens.
        for new_id, pair in enumerate(
            self.merges, self.FIRST_MERGE_ID
        ):
            if len(pair) != 2 or any(
                not isinstance(part, int)
                or part not in self.id_to_bytes
                for part in pair
            ):
                raise ValueError(
                    f"Invalid merge rule: {pair}"
                )

            self.id_to_bytes[new_id] = (
                self.id_to_bytes[pair[0]]
                + self.id_to_bytes[pair[1]]
            )

    @property
    def vocab_size(self):
        return (
            self.FIRST_MERGE_ID + len(self.merges)
        )

    @staticmethod
    def _merge_pair(tokens, pair, new_id):
        result = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i + 1]) == pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    @classmethod
    def train(cls, text, target_vocab_size=320):
        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(
                "Training text must be nonempty"
            )

        if target_vocab_size < cls.FIRST_MERGE_ID:
            raise ValueError(
                "Target vocabulary must be at least 258"
            )

        tokens = list(text.encode("utf-8"))
        merges = []

        for new_id in range(
            cls.FIRST_MERGE_ID,
            target_vocab_size
        ):
            counts = Counter(
                zip(tokens, tokens[1:])
            )

            if not counts:
                break

            # Prefer the most frequent pair.
            # Resolve ties deterministically.
            pair, frequency = max(
                counts.items(),
                key=lambda entry: (
                    entry[1],
                    -entry[0][0],
                    -entry[0][1],
                )
            )

            if frequency < 2:
                break

            tokens = cls._merge_pair(
                tokens, pair, new_id
            )

            merges.append(pair)

        return cls(merges)

    def encode(self, text, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("Input must be text")

        tokens = list(text.encode("utf-8"))

        for new_id, pair in enumerate(
            self.merges, self.FIRST_MERGE_ID
        ):
            tokens = self._merge_pair(
                tokens, pair, new_id
            )

        if add_eos:
            tokens.append(self.EOS_ID)

        return tokens

    def decode(self, token_ids, skip_special=True):
        chunks = []

        for token in token_ids:
            token = int(token)

            if token in (
                self.PAD_ID, self.EOS_ID
            ):
                if not skip_special:
                    raise ValueError(
                        "Special tokens cannot be "
                        "converted to UTF-8 text"
                    )
                continue

            if token not in self.id_to_bytes:
                raise ValueError(
                    f"Unknown token ID: {token}"
                )

            chunks.append(
                self.id_to_bytes[token]
            )

        return b"".join(chunks).decode("utf-8")

    def fingerprint(self):
        encoded = json.dumps(
            self.merges,
            separators=(",", ":")
        ).encode("utf-8")

        return hashlib.sha256(
            encoded
        ).hexdigest()

    def save(self, path):
        path = Path(path)

        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        payload = {
            "format_version": 1,
            "type": "skygrid_byte_bpe",
            "pad_id": self.PAD_ID,
            "eos_id": self.EOS_ID,
            "merges": [
                list(pair) for pair in self.merges
            ],
            "fingerprint": self.fingerprint(),
        }

        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get("format_version") != 1
            or payload.get("type") != "skygrid_byte_bpe"
            or payload.get("pad_id") != cls.PAD_ID
            or payload.get("eos_id") != cls.EOS_ID
        ):
            raise ValueError(
                "Incompatible SkyGrid BPE tokenizer file"
            )

        tokenizer = cls(payload["merges"])

        if (
            tokenizer.fingerprint()
            != payload.get("fingerprint")
        ):
            raise ValueError(
                "Tokenizer merge fingerprint mismatch"
            )

        return tokenizer
