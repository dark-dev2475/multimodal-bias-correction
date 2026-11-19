"""Download one pinned free text NLI model (~579 MB); no credentials or training."""
import hashlib
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'cross-encoder/nli-deberta-v3-small'
REVISION = 'fa2804872c3b4bd748f38c0185cc85775361e735'
OUT = ROOT / 'data/local_vlm/nli-deberta-v3-small'
FILES = {
    'config.json': (1052, None), 'model.safetensors': (567605820, 'ebc79588dd73ccfb6a3f6078519cfbf512c5305384c5ea1845bc71cd32216e86'),
    'special_tokens_map.json': (301, None),
    'spm.model': (2464616, 'c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd'),
    'tokenizer.json': (8656624, None), 'tokenizer_config.json': (1346, None),
}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, (size, expected_sha) in FILES.items():
        target = OUT / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.partial')
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                if offset and response.status_code == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected download range')
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as stream:
                    for block in response.iter_content(1024*1024):
                        if block:
                            stream.write(block)
            if partial.stat().st_size != size or (expected_sha and sha(partial) != expected_sha):
                raise ValueError('Downloaded file failed integrity check: ' + name)
            partial.replace(target)
        actual_sha = sha(target)
        if target.stat().st_size != size or (expected_sha and actual_sha != expected_sha):
            raise ValueError('Installed file failed integrity check: ' + name)
        verified[name] = {'bytes': size, 'sha256': actual_sha}
        print('Verified ' + name, flush=True)
    metadata = {'model_id': MODEL_ID, 'revision': REVISION, 'files': verified,
                'license': 'Apache-2.0', 'purpose': 'Separate text-entailment validation, not a vision model',
                'model_card': 'https://huggingface.co/' + MODEL_ID}
    (OUT / 'installation.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
