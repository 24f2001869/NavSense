"""
SIH26168 - Git LFS Fetcher for IO-VNBD
Utility to resolve and download genuine CSV payload files from GitHub Git-LFS
storage for specific trips or datasets without modifying raw file paths.
"""

import os
import sys
from pathlib import Path
import requests

LFS_BATCH_URL = "https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch"

def parse_pointer_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]
    
    oid = None
    size = None
    for line in lines:
        if line.startswith("oid sha256:"):
            oid = line.replace("oid sha256:", "").strip()
        elif line.startswith("size"):
            parts = line.split()
            if len(parts) >= 2:
                size = int(parts[1])
    return oid, size

def download_lfs_file(pointer_path, output_path=None, force=False):
    pointer_path = Path(pointer_path)
    if output_path is None:
        output_path = pointer_path
    else:
        output_path = Path(output_path)

    oid, size = parse_pointer_file(pointer_path)
    if not oid or not size:
        print(f"[SKIP] Not an LFS pointer: {pointer_path}")
        return False

    if output_path.exists() and output_path.stat().st_size == size and not force:
        print(f"[EXISTS] Already downloaded ({size:,} bytes): {output_path.name}")
        return True

    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}]
    }
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json"
    }

    resp = requests.post(LFS_BATCH_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"[ERROR] Batch request failed ({resp.status_code}): {resp.text}")
        return False

    data = resp.json()
    objects = data.get("objects", [])
    if not objects or "actions" not in objects[0] or "download" not in objects[0]["actions"]:
        print(f"[ERROR] No download URL returned for {pointer_path.name}")
        return False

    download_url = objects[0]["actions"]["download"]["href"]
    print(f"[DOWNLOADING] {pointer_path.name} ({size / (1024*1024):.2f} MB)...")

    temp_output = output_path.with_suffix(".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(temp_output, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    temp_output.replace(output_path)
    print(f"[SUCCESS] Downloaded: {output_path.name} ({output_path.stat().st_size:,} bytes)")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        download_lfs_file(target)
    else:
        print("Usage: python download_lfs.py <pointer_file_path>")
