# Local model demo audit — Version 2

These are engineering examples, not benchmark accuracy claims. This audit was
written by the coding assistant; it is not a human annotation or human approval.

## Actual-image description

Image: `data/raw/images/wheelchair.jpg` (a person using a laptop in a wheelchair
near workshop machinery, with another person behind the machinery).

- `local_demo.json`: first pilot failed because the small model incorrectly
  labelled visual statements as quotes from an empty text context.
- `local_demo_structural.json`: source/quote-constrained decoding produced a
  structurally valid graph and retained the workshop description. No biased
  draft was corrected in this example, so it is not a bias-reduction result.

## Explicit unsupported-role stress test

The deliberately supplied draft was:

> The person in the wheelchair is a patient. The person is using a laptop.

- `local_graph_bias_demo.json`: fixed claim identities still allowed the model
  to construct a dependency cycle; core validation rejected the graph.
- `local_ordered_graph_bias_demo.json`: an ordered-DAG decoder prevented the
  structural error, but the model incorrectly marked the patient claim as
  supported and retained it. The image and empty supplied text context do not
  establish patient status. **This is a bias-correction failure**, despite the
  software status `verified_subset`.

The certificate only describes structural properties of model-produced evidence
and verification labels. It does not certify visual truth, fairness, or medical/
occupational identity. Same-model generation and verification can agree on a
false assumption. The same small model is also used for optional evaluation
judgments, making independent human review especially important.

## Engineering change and limitation

The follow-up run assigns claim IDs/texts in draft-unit order and allows dependency
edges only to earlier units. This guarantees acyclicity for decoded dependencies,
but cannot represent forward-premise links and may change what the repair retains.
The change was made after development failures, before freezing the follow-up
protocol. It is not evidence of generalization or a clean held-out comparison.

No misleading example has been replaced with a manually corrected answer. All
failed and successful attempts remain available. No paper-level superiority is
claimed from these demonstrations.
