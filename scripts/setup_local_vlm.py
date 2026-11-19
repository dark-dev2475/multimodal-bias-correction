"""Download pinned official Qwen GGUF + llama.cpp Vulkan runtime, verify SHA256.

User-approved local installation; no paid services, administrator changes or drivers.
"""
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/local_vlm"
REVISION = "52d6c8ffea26cc873ac5ad116f8631268d7eb503"
BASE = f"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/{REVISION}/"
FILES = [
    ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", BASE + "Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1107409952,
     "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"),
    ("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", BASE + "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", 445053216,
     "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"),
    ("llama-b12098-bin-win-vulkan-x64.zip", "https://github.com/ggml-org/llama.cpp/releases/download/b12098/llama-b12098-bin-win-vulkan-x64.zip", 37660090,
     "3d8f6c95b0a18c9b3c7bdfc36cf22400d3f0c03551a89e72f7b90dc42bac5e2a"),
]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(name, url, size, expected):
    path = DEST / name
    if path.exists():
        if path.stat().st_size != size or sha(path) != expected:
            raise ValueError("Existing file checksum mismatch; refusing overwrite: " + name)
        return path
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise ValueError("Partial download larger than pinned asset")
        try:
            with requests.get(url, stream=True, timeout=(30, 45), headers={"Range": f"bytes={offset}-"} if offset else {}) as r:
                r.raise_for_status()
                if offset and r.status_code == 206 and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid resumed byte range")
                append = offset > 0 and r.status_code == 206
                written = offset if append else 0
                tick = time.monotonic()
                with part.open("ab" if append else "wb") as handle:
                    for chunk in r.iter_content(1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                        if written > size:
                            raise ValueError("Downloaded bytes exceed pinned size")
                        if time.monotonic() - tick > 15:
                            print(f"{name}: {100*written/size:.1f}%", flush=True)
                            tick = time.monotonic()
                if written == size:
                    break
        except requests.RequestException as exc:
            print(f"Download interrupted ({type(exc).__name__}); preserving partial {name}", flush=True)
            if attempt == 3:
                raise RuntimeError("Download unavailable; rerun to resume") from None
            time.sleep(1)
    if part.stat().st_size != size or sha(part) != expected:
        raise ValueError("Download size/checksum failed: " + name)
    part.replace(path)
    print("Verified " + name, flush=True)
    return path


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    needed = sum(size for name, _, size, _ in FILES if not (DEST / name).exists())
    if shutil.disk_usage(DEST).free < needed + 2_000_000_000:
        raise RuntimeError("Insufficient disk space with 2 GB safety reserve")
    for row in FILES:
        download(*row)
    runtime = DEST / "runtime"
    runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(DEST / FILES[-1][0]) as archive:
        for member in archive.infolist():
            target = (runtime / member.filename).resolve()
            if not target.is_relative_to(runtime.resolve()):
                raise ValueError("Unsafe archive path")
        archive.extractall(runtime)
    metadata = {"model": "Qwen/Qwen3-VL-2B-Instruct-GGUF", "model_revision": REVISION,
                "runtime": "llama.cpp b12098 Vulkan Windows x64 (prerelease build)",
                "model_alias": "qwen3-vl-2b-local", "endpoint": "http://127.0.0.1:8087/v1",
                "verified_files": [{"name": n, "bytes": s, "sha256": h, "url": u} for n, u, s, h in FILES]}
    (DEST / "installation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Local model and runtime ready. No model server started by this installer.", flush=True)


if __name__ == "__main__":
    main()
