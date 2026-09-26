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
    },
    "alpaca_lora_7b": {
        "url": "https://huggingface.co/tloen/alpaca-lora-7b/resolve/main/adapter_model.bin",
        "dest": "adapters/clean/alpaca_lora_7b/adapter_model.bin",
        "description": "Stanford Alpaca LoRA 7B adapter weights (~64 MB)"
    },
    "mistral7b_q4km": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "dest": "models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral-7B-Instruct-v0.2 Q4_K_M GGUF (~4.07 GB)"
    }
}

def download_file(url: str, dest_path: str, desc: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    max_retries = 20
    retry_count = 0
    total_size = 0

    print(f"[START] Downloading {desc}")
    print(f"        Destination: {dest_path}")

    while retry_count < max_retries:
        existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        if existing_bytes > 0:
            print(f"[RESUME] Resuming from {existing_bytes / (1024*1024):.1f} MB ({existing_bytes / (1024**3):.2f} GB)...")

        headers = {"User-Agent": "CALB-Shield-Empirical-Runner"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(url, headers=headers)
        ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                cr = resp.headers.get("Content-Range")
                if cr:
                    # Content-Range format: bytes start-end/total
                    try:
                        total_size = int(cr.split("/")[-1])
                    except (ValueError, IndexError):
                        pass
                if total_size == 0:
                    cl = resp.headers.get("Content-Length")
                    if cl is not None:
                        total_size = int(cl) + existing_bytes

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
                            print(f"  [PROG] {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({pct:.1f}%)", flush=True)
                            last_log_time = now

            if total_size > 0 and downloaded < total_size:
                raise IOError(f"Incomplete download: {downloaded} of {total_size} bytes received. Will retry.")

            os.rename(temp_path, dest_path)
            print(f"[SUCCESS] Download completed: {dest_path}")
            return
        except Exception as e:
            retry_count += 1
            print(f"[WARN] Connection dropped: {e}. Auto-retrying ({retry_count}/{max_retries}) in 3s...", flush=True)
            time.sleep(3)

    print(f"[ERROR] Max retries ({max_retries}) reached. Download failed.")
    sys.exit(1)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "llama3_q4km"
    if target not in MODELS:
        print(f"Unknown target: {target}. Available: {list(MODELS.keys())}")
        sys.exit(1)
    cfg = MODELS[target]
    download_file(cfg["url"], cfg["dest"], cfg["description"])
