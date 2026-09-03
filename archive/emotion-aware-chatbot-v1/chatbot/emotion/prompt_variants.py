"""Built-in emotion prompt variants for ablation experiments.

The ``full`` variant is the application default and keeps using the
configurable ``DEFAULT_EMOTION_ANALYSIS_PROMPT`` (which may be overridden via
``PROMPT_CONFIG_PATH``).  The four experiment variants are fixed templates that
share the same placeholders and response constraints; they exist only for
controlled prompt experiments and never change the default behaviour.
"""

from chatbot.core.prompt_config import DEFAULT_EMOTION_ANALYSIS_PROMPT

DEFAULT_PROMPT_VARIANT = "full"
PROMPT_VARIANT_NAMES = frozenset({
    "full",
    "prompt_no_label_guidance",
    "prompt_concise_direct",
    "prompt_coarse_to_fine",
    "prompt_contrastive_check",
})

# Shared response constraints for experiment variants.  Variants must not alter
# the output schema or value restrictions on their own.
COMMON_RESPONSE_BLOCK = """- Response Format: Return exactly one JSON object with these fields:
  {"primary_emotion": "anxious", "confidence": 0.0, "secondary_emotions": [], "evidence": "short phrase from the dialogue", "reply_strategy": "brief guidance for the next chatbot reply", "trajectory_note": "optional change from prior emotion", "safety_level": "normal"}
  Use primary_emotion and secondary_emotions only from the provided Emotion labels. Use safety_level as one of: normal, supportive, crisis."""

PROMPT_NO_LABEL_GUIDANCE = """Infer the emotion expressed by the target user in the described situation or current input.
- Dialogue context: The conversation history between user and assistant, with utterances separated by </s>.
- Emotion labels: {emotion_labels}
- Choose a single inferred emotion from the provided Emotion labels, not outside of them.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_CONCISE_DIRECT = """Select exactly one emotion label that best matches the target user's current input.
- Emotion labels: {emotion_labels}
- Use the label definitions and examples as evidence. Do not choose a label outside the list.
- Label definitions:
{label_guidance}
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_COARSE_TO_FINE = """Infer the emotion expressed by the target user in the described situation or current input.
- Emotion labels: {emotion_labels}
- Label definitions:
{label_guidance}
- First identify the broad emotion family internally, then select the single most precise label from the provided list.
- Return only the final structured result. Do not reveal the intermediate family decision.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_CONTRASTIVE_CHECK = """Infer the emotion expressed by the target user in the described situation or current input.
- Emotion labels: {emotion_labels}
- Label definitions:
{label_guidance}
- Compare the two most plausible labels internally against the dialogue evidence and label boundaries before choosing one.
- Return only the final structured result. Do not reveal the candidate comparison.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

BUILTIN_PROMPT_VARIANTS = {
    "full": DEFAULT_EMOTION_ANALYSIS_PROMPT,
    "prompt_no_label_guidance": PROMPT_NO_LABEL_GUIDANCE,
    "prompt_concise_direct": PROMPT_CONCISE_DIRECT,
    "prompt_coarse_to_fine": PROMPT_COARSE_TO_FINE,
    "prompt_contrastive_check": PROMPT_CONTRASTIVE_CHECK,
}


def resolve_emotion_prompt_template(prompt_variant: str) -> str:
    try:
        return BUILTIN_PROMPT_VARIANTS[prompt_variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown emotion prompt variant: {prompt_variant!r}"
        ) from exc
