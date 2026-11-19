"""Post-hoc, offline text-similarity diagnostic; never part of inference.

Uses an already cached MiniLM model. Cosine similarity is NOT accuracy, factual
entailment, or fairness. Failed/missing answers receive no fabricated score.
"""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/version2/FINAL_40_IMAGE_STUDY'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots' / REVISION
SYSTEMS = ('raw', 'eigr', 'self_refine_vqa', 'intent_aware_vqa', 'reflexion_vqa',
           'constitutional_critique_vqa', 'always_abstain')
WARNING = ('Exploratory post-hoc text similarity, not accuracy/fairness/entailment. '
           'Negated or biased sentences can be highly similar. References are '
           'AI-assisted drafts pending human review; the 40 cases are development data.')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def comparison_rows(bundle):
    rows = []
    seen = set()
    for example in bundle['examples']:
        case = example['case']
        cid = case['example_id']
        if cid in seen:
            raise ValueError('Duplicate case: ' + cid)
        seen.add(cid)
        reference = example['reference']['reference']['reference_answer']
        for system in SYSTEMS:
            result = example['outputs'].get(system, {})
            status = result.get('status', 'not_collected')
            answer = result.get('final_response')
            eligible = status not in {'failed', 'not_collected'} and bool(answer and answer.strip())
            rows.append({'example_id': cid, 'system': system, 'status': status,
                         'answer': answer, 'reference_answer': reference,
                         'scorable': eligible, 'cosine_similarity': None})
    return rows


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError('Embedding dimensions differ')
    denominator = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(y) ** 2 for y in b))
    if denominator == 0:
        raise ValueError('Zero embedding has undefined cosine similarity')
    return max(-1.0, min(1.0, sum(float(x) * float(y) for x, y in zip(a, b)) / denominator))


def summarize(rows):
    summary = {}
    for system in SYSTEMS:
        selected = [r for r in rows if r['system'] == system]
        scores = [r['cosine_similarity'] for r in selected if r['cosine_similarity'] is not None]
        summary[system] = {'planned': len(selected), 'scored': len(scores),
                           'unscored': len(selected) - len(scores),
                           'mean_cosine_scored_only': sum(scores) / len(scores) if scores else None}
    return summary


def main(output, model_path):
    # Explicitly forbid network fallback; no download and no API key are needed.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if not (model_path / 'model.safetensors').is_file():
        raise FileNotFoundError('Supply a complete local MiniLM snapshot; no download is attempted.')
    source = output / 'all_outputs.json'
    bundle = json.loads(source.read_text(encoding='utf-8'))
    rows = comparison_rows(bundle)
    texts = sorted({r[key] for r in rows if r['scorable'] for key in ('answer', 'reference_answer')})
    if not texts:
        raise ValueError('No valid answer/reference pairs')
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(2)
    model = SentenceTransformer(str(model_path), device='cpu', local_files_only=True,
                                trust_remote_code=False)
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    embeddings = dict(zip(texts, vectors))
    lengths = {text: len(model.tokenizer(text, truncation=False)['input_ids']) for text in texts}
    for row in rows:
        if row['scorable']:
            row['cosine_similarity'] = cosine(embeddings[row['answer']], embeddings[row['reference_answer']])
            row['answer_tokens_before_truncation'] = lengths[row['answer']]
            row['reference_tokens_before_truncation'] = lengths[row['reference_answer']]
            row['text_truncated'] = any(lengths[row[key]] > model.max_seq_length
                                        for key in ('answer', 'reference_answer'))
    metadata = {'warning': WARNING, 'evaluation_only': True, 'post_hoc': True,
                'source_bundle_sha256': file_sha256(source), 'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
                'model_snapshot_directory': str(model_path.resolve()), 'model_revision_directory': model_path.name,
                'model_file_sha256': {str(p.relative_to(model_path)): file_sha256(p)
                                      for p in sorted(model_path.rglob('*')) if p.is_file()},
                'max_sequence_length': model.max_seq_length, 'embedding_dimension': model.get_sentence_embedding_dimension(),
                'network_used': False, 'training_performed': False,
                'generated_at': datetime.now(timezone.utc).isoformat(), 'systems': summarize(rows)}
    all_methods = {**metadata, 'results': rows}
    eigr = {**metadata, 'results': [r for r in rows if r['system'] == 'eigr']}
    for name, payload in [('embedding_similarity_all_methods.json', all_methods), ('embedding_similarity_40.json', eigr)]:
        (output / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = ['# Offline embedding similarity — exploratory diagnostic', '', WARNING, '',
              'No training or network requests. A cached MiniLM encoder compares final answer text to reference text; '
              'it does not inspect the image. This analysis was added after collection began and is not a preregistered endpoint.', '',
              '| Method | Scored / planned | Mean cosine (scored only) |', '|---|---:|---:|']
    for system, summary in metadata['systems'].items():
        value = summary['mean_cosine_scored_only']
        score = 'not available' if value is None else f'{value:.4f}'
        report.append(f"| {system} | {summary['scored']}/{summary['planned']} | {score} |")
    report += ['', 'Cosine is not a percentage of correct responses. Failed/missing outputs are unscored, never '
               'assigned a successful score. Compare coverage before interpreting averages. Long text may be '
               'truncated at the encoder limit; per-response flags record this.', '',
               'Model: [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). '
               'Model-file and input hashes are saved in the JSON exports.']
    (output / 'EMBEDDING_SIMILARITY.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(metadata['systems'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model-path', type=Path, default=MODEL)
    args = parser.parse_args()
    main(args.output, args.model_path)
