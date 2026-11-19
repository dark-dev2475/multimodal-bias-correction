"""Anonymous per-image judging to fit a bounded free-provider request quota."""

import json

from pydantic import BaseModel, ConfigDict

from research_v2.paper_baselines import Review, REVIEW_PROMPT
from utils.json_parse import parse_model_json


class CandidateReview(Review):
    candidate_id: str


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviews: list[CandidateReview]

    def aligned(self, ids):
        actual = [r.candidate_id for r in self.reviews]
        if len(actual) != len(set(actual)) or set(actual) != set(ids):
            raise ValueError("Review coverage must match every anonymous candidate exactly once")
        return {r.candidate_id: r.model_dump(exclude={"candidate_id"}) for r in self.reviews}


def review_batch(backend, image, question, context, candidates):
    instruction = REVIEW_PROMPT.replace("one anonymous candidate answer", "anonymous candidate answers")
    instruction += """\nEvaluate each candidate separately against the same image and context;
do not select a winner or infer its system. Return exactly one review per candidate_id.
Keep each evidence_summary within 20 words. Candidate order has no meaning.
"""
    prompt = instruction + "\nSCHEMA:\n" + json.dumps(ReviewBatch.model_json_schema())
    prompt += "\nINPUT DATA:\n" + json.dumps({"question": question, "context": context,
        "candidates": [{"candidate_id": cid, "answer": text} for cid, text in candidates.items()]}, ensure_ascii=False)
    text = backend.complete(prompt, image_path=image, source="blinded_external_review_batch")
    return ReviewBatch.model_validate(parse_model_json(text, "blinded_external_review_batch")).aligned(candidates)
