"""Bounded parallel range download of the SAME checksum-pinned official binary.

Fallback for interrupted GitHub large transfers. No mirrors or TLS bypasses.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from setup_local_vlm import DEST, FILES, sha


def main():
    name, _, size, expected = FILES[-1]
    target = DEST / name
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing runtime checksum mismatch')
        return
    chunks = DEST / 'runtime_chunks'
    chunks.mkdir(exist_ok=True)
    response = requests.get('https://api.github.com/repos/ggml-org/llama.cpp/releases/assets/708130753',
        headers={'Accept': 'application/octet-stream'}, allow_redirects=False, timeout=(20, 20))
    if response.status_code != 302:
        raise RuntimeError(f'Expected official asset redirect, got {response.status_code}')
    url = response.headers['Location']
    if not url.startswith('https://release-assets.githubusercontent.com/'):
        raise ValueError('Unexpected official asset host')
    block = 262144
    ranges = [(start, min(size-1, start+block-1)) for start in range(0, size, block)]

    def fetch(bounds):
        start, end = bounds
        path = chunks / f'{start:010d}.bin'
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        for attempt in range(3):
            try:
                r = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 20))
                if r.status_code == 429:
                    raise RuntimeError('CDN rate limited; stop and resume later')
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Server did not honor exact range')
                if len(r.content) != end-start+1:
                    raise ValueError('Incomplete segment')
                path.write_bytes(r.content)
                return path
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError('Official asset connection interrupted; segments preserved') from None
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, bounds) for bounds in ranges]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 8 == 0 or completed == len(ranges):
                print(f'Runtime segments: {completed}/{len(ranges)}', flush=True)
    temporary = target.with_suffix('.parallel.part')
    with temporary.open('wb') as handle:
        for start, _ in ranges:
            handle.write((chunks / f'{start:010d}.bin').read_bytes())
    if temporary.stat().st_size != size or sha(temporary) != expected:
        raise ValueError('Assembled official runtime checksum mismatch')
    temporary.replace(target)
    print('Official runtime SHA256 verified.', flush=True)


if __name__ == '__main__':
    main()
