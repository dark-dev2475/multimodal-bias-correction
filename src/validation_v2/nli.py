"""Independent frozen text entailment, not visual truth or a bias classifier."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'data/local_vlm/nli-deberta-v3-small'
LABELS = ['contradicted', 'entailed', 'unknown']  # Official model-card label order.


class TextNLI:
    def __init__(self):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True,
                                                                       trust_remote_code=False)
        self.model.eval()
        if self.model.config.num_labels != 3:
            raise ValueError('Expected official three-label NLI model')

    def predict(self, premise, hypothesis):
        features = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=False)
        length = features['input_ids'].shape[1]
        if length > 512:
            raise ValueError('NLI pair exceeds 512 tokens; no silent truncation allowed')
        with self.torch.inference_mode():
            logits = self.model(**features).logits[0]
            values = self.torch.softmax(logits, dim=-1).tolist()
        return {'prediction': LABELS[max(range(3), key=lambda i: values[i])],
                'raw_softmax_not_calibrated_probabilities': dict(zip(LABELS, values)),
                'input_tokens': length}
