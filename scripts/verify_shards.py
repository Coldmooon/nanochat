#!/usr/bin/env python
import argparse
import hashlib
import os
import sys
import textwrap

import requests

"""
Example:

python verify_shards.py \
    --repo karpathy/fineweb-edu-100b-shuffle \
    --local-dir ~/.cache/nanochat/base_data \
    --start 0 --end 1821

"""


def fetch_remote_sha256(repo_id: str, filename: str, session: requests.Session) -> str:
    """
    Parse sha256 from HuggingFace raw pointer files.
    Example:
    https://huggingface.co/datasets/{repo_id}/raw/main/{filename}
    Content looks like:
    version https://git-lfs.github.com/spec/v1 oid sha256:... size 123456
    """
    # Allow overriding the endpoint via HF_ENDPOINT env var, defaults to hf-mirror.
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    url = f"{hf_endpoint}/datasets/{repo_id}/raw/main/{filename}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()

    # Simple parser: locate the token that starts with 'sha256:'
    for part in text.split():
        if part.startswith("sha256:"):
            return part.split("sha256:")[1]

    raise RuntimeError(f"Cannot find sha256 in pointer file for {filename}. "
                       f"Pointer content: {text!r}")


def compute_local_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Stream the file to compute sha256 without loading very large files into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify HF Xet-backed shards against official sha256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example usage:

              # verify the first 13 shards (00000 ~ 00012)
              python verify_shards.py \\
                  --repo karpathy/fineweb-edu-100b-shuffle \\
                  --local-dir ~/.cache/nanochat/base_data \\
                  --start 0 --end 12

              # verify the first 1822 shards (00000 ~ 01821)
              python verify_shards.py \\
                  --repo karpathy/fineweb-edu-100b-shuffle \\
                  --local-dir ~/.cache/nanochat/base_data \\
                  --start 0 --end 1821
            """
        ),
    )
    parser.add_argument("--repo", required=True,
                        help="HuggingFace dataset id, e.g. karpathy/fineweb-edu-100b-shuffle")
    parser.add_argument("--local-dir", required=True,
                        help="Local shard directory, e.g. ~/.cache/nanochat/base_data")
    parser.add_argument("--start", type=int, default=0,
                        help="Inclusive start shard index, e.g. 0 for shard_00000.parquet")
    parser.add_argument("--end", type=int, required=True,
                        help="Inclusive end shard index, e.g. 12 for shard_00012.parquet")
    parser.add_argument("--pattern", default="shard_{:05d}.parquet",
                        help="Filename pattern, default shard_{:05d}.parquet")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    if not os.path.isdir(local_dir):
        print(f"[ERROR] Local dir not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    mismatches = []
    missing = []
    remote_errors = []

    for idx in range(args.start, args.end + 1):
        filename = args.pattern.format(idx)
        local_path = os.path.join(local_dir, filename)
        print(f"\n=== Checking {filename} ===")

        # 1. Remote sha256
        try:
            remote_sha = fetch_remote_sha256(args.repo, filename, session)
            print(f"  remote sha256: {remote_sha}")
        except Exception as e:
            print(f"  [REMOTE ERROR] {e}")
            remote_errors.append((filename, str(e)))
            continue

        # 2. Check local file exists
        if not os.path.exists(local_path):
            print(f"  [MISSING] local file not found: {local_path}")
            missing.append(filename)
            continue

        # 3. Local sha256
        try:
            local_sha = compute_local_sha256(local_path)
            print(f"  local  sha256: {local_sha}")
        except Exception as e:
            print(f"  [LOCAL ERROR] failed to hash {local_path}: {e}")
            mismatches.append(filename)
            continue

        # 4. Compare
        if local_sha == remote_sha:
            print(f"  [OK] {filename} ✅")
        else:
            print(f"  [MISMATCH] {filename} ❌")
            mismatches.append(filename)

    # Summary
    print("\n====== SUMMARY ======")
    print(f"Checked indices: {args.start} .. {args.end}")
    print(f"Missing   files: {len(missing)}")
    print(f"Mismatches    : {len(mismatches)}")
    print(f"Remote errors : {len(remote_errors)}")

    if missing:
        print("\nMissing files:")
        for f in missing:
            print("  -", f)

    if mismatches:
        print("\nMismatched files:")
        for f in mismatches:
            print("  -", f)

    if remote_errors:
        print("\nRemote errors:")
        for f, e in remote_errors:
            print(f"  - {f}: {e}")


if __name__ == "__main__":
    main()

