"""Explicit, versioned principles used by the proposed corrector."""

CORRECTION_POLICY_VERSION = "v1"

CORRECTION_PRINCIPLES = (
    "Only state information visually supported by the image.",
    "Do not infer medical status from appearance or assistive devices alone.",
    "Do not infer occupation from workplace, clothing, tools, or context alone.",
    "Do not infer socioeconomic status from clothing, possessions, or environment.",
    "Do not infer relationships from proximity or interaction alone.",
    "Do not infer emotions or intentions unless visually well supported.",
    "Do not infer demographic or identity attributes unless visually established and relevant.",
    "Preserve supported visual observations.",
    "Do not introduce new facts during correction.",
    "Remove or rewrite only unsupported or problematic claims.",
    "Prefer uncertainty over unsupported inference.",
    "Keep the corrected response natural and informative.",
)


def policy_text() -> str:
    return "\n".join(
        f"{index}. {principle}"
        for index, principle in enumerate(CORRECTION_PRINCIPLES, start=1)
    )
