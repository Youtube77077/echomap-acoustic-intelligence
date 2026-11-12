"""Fetch the labelled test recordings into a directory (CI only).  Usage: python tools/fetch_test_audio.py <dest>

Every clip has a known ground-truth label from its source dataset (see tests/test_audio_manifest.json).
Recordings are never committed to git.
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "test_audio_manifest.json"


def main():
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "EchoMap-CI"
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]:
        p = dest / item["file"]
        if not p.exists():
            r = s.get(item["url"], timeout=300)
            r.raise_for_status()
            p.write_bytes(r.content)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if item.get("sha256") and h != item["sha256"]:
            sys.exit(f"CHECKSUM MISMATCH {item['file']}")
        print("ok", item["file"])


if __name__ == "__main__":
    main()
