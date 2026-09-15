#!/usr/bin/env python3
"""
download_models.py
Resumable chunked downloader for CALB-Shield empirical models.
"""

import os
import sys
import time
import urllib.request
import ssl

MODELS = {
    "llama3_q4km": {
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "dest": "models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "description": "Meta Llama-3-8B-Instruct Q4_K_M GGUF (~4.58 GB)"
    }
}

def download_file(url: str, dest_path: str, desc: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    existing_bytes = 0
    if os.path.exists(dest_path):
        print(f"[EXISTS] {dest_path} already exists. Skipping download.")
        return

    if os.path.exists(temp_path):
        existing_bytes = os.path.getsize(temp_path)
        print(f"[RESUME] Resuming {desc} from {existing_bytes / (1024*1024):.1f} MB...")

    headers = {"User-Agent": "CALB-Shield-Empirical-Runner"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            total_size = resp.headers.get("Content-Length")
            if total_size is not None:
                total_size = int(total_size) + existing_bytes
            else:
                total_size = 0

            print(f"[START] Downloading {desc}")
            print(f"        Destination: {dest_path}")
            if total_size > 0:
                print(f"        Total Size:  {total_size / (1024**3):.2f} GB")

            mode = "ab" if existing_bytes > 0 else "wb"
            downloaded = existing_bytes
            last_log_time = time.time()
            chunk_size = 1024 * 1024 * 4  # 4 MB chunks

            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log_time >= 5.0:
                        pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        speed = (len(chunk) / (now - last_log_time + 1e-5)) / (1024 * 1024)
                        print(f"  [PROG] {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({pct:.1f}%)", flush=True)
                        last_log_time = now

        os.rename(temp_path, dest_path)
        print(f"[SUCCESS] Download completed: {dest_path}")
    except Exception as e:
        print(f"[ERROR] Download interrupted: {e}")
        sys.exit(1)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "llama3_q4km"
    if target not in MODELS:
        print(f"Unknown target: {target}. Available: {list(MODELS.keys())}")
        sys.exit(1)
    cfg = MODELS[target]
    download_file(cfg["url"], cfg["dest"], cfg["description"])
