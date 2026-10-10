
"""Extract PX4 Markdown documents with provenance."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    ROOT / "data" / "aviation"
    / "source" / "px4_user_guide"
)

OUTPUT = (
    ROOT / "data" / "aviation" / "processed_v09"
)

SECTIONS = (
    "concept",
    "data_links",
    "flight_modes",
)

LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def normalize_markdown(raw):
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    raw = re.sub(
        r"<!--.*?-->",
        "",
        raw,
        flags=re.DOTALL,
    )

    raw = re.sub(
        r"!\[[^\]]*\]\([^\n]*?\)",
        "",
        raw,
    )

    raw = re.sub(
        r"(?<!!)\[([^\]]+)\]\([^\n]*?\)",
        r"\1",
        raw,
    )

    return re.sub(
        r"\n{3,}", "\n\n", raw
    ).strip()


def ingest_documents(repo, target, revision):
    repo = Path(repo)
    target = Path(target)

    manifest = []
    seen = set()

    (target / "docs").mkdir(
        parents=True,
        exist_ok=True,
    )

    for section in SECTIONS:
        folder = repo / "en" / section

        for source in sorted(folder.glob("*.md")):
            relative = source.relative_to(
                repo
            ).as_posix()

            raw = source.read_text(
                encoding="utf-8-sig"
            )

            cleaned = normalize_markdown(raw)

            if len(cleaned) < 350:
                continue

            digest = hashlib.sha256(
                cleaned.encode("utf-8")
            ).hexdigest()

            if digest in seen:
                continue

            seen.add(digest)

            output_name = (
                hashlib.sha256(
                    relative.encode("utf-8")
                ).hexdigest()[:16]
                + ".txt"
            )

            (target / "docs" / output_name).write_text(
                cleaned,
                encoding="utf-8",
            )

            manifest.append({
                "file": output_name,
                "source_path": relative,
                "source_url": (
                    "https://github.com/"
                    "PX4/PX4-user_guide/blob/"
                    f"{revision}/{relative}"
                ),
                "original_project": "PX4 User Guide",
                "documentation_license": "CC BY 4.0",
                "license_url": LICENSE_URL,
                "rights_review": (
                    "required before model training"
                ),
                "sha256": digest,
                "characters": len(cleaned),
            })

    if not manifest:
        raise ValueError(
            "No documents found. Check sparse checkout."
        )

    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def main():
    if not (SOURCE / ".git").exists():
        raise FileNotFoundError(
            "Clone the PX4 User Guide first."
        )

    revision = subprocess.check_output(
        [
            "git",
            "-C",
            str(SOURCE),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    destination = OUTPUT / revision[:12]

    rows = ingest_documents(
        SOURCE,
        destination,
        revision,
    )

    print(
        f"Imported {len(rows)} documents for review"
    )

    print(
        "Characters:",
        sum(row["characters"] for row in rows),
    )

    print(f"Source revision: {revision}")
    print(f"Local corpus: {destination}")
    print(
        "Document quality and training rights "
        "still require review."
    )


if __name__ == "__main__":
    main()
