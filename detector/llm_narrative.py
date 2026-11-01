"""LLM-powered forensic narrative generator — v4.0 (NEW).

Generates intelligent, context-aware forensic explanations using
an LLM (OpenAI GPT-4o or Google Gemini) as an optional enhancement.

When an API key is configured, the LLM is given:
  * The verdict and P(fake)
  * All evidence with logits and reliabilities
  * Quality profile (resolution, compression, face presence)
  * Attribution (generator family if identified)
  * Stability results

It produces a professional, plain-language forensic summary that:
  * Explains WHY the verdict was reached
  * Describes which specific signals drove the decision
  * Notes limitations and uncertainties
  * Is appropriate for a technical forensic report

Fallback: if no LLM is configured or the API call fails, the existing
rule-based narrative generator is used automatically.

Provider selection (in .env):
    LLM_NARRATIVE_PROVIDER=openai    # use OpenAI
    LLM_NARRATIVE_PROVIDER=gemini    # use Google Gemini
    LLM_NARRATIVE_PROVIDER=none      # rule-based only (default)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("deepfake.llm_narrative")

_SYSTEM_PROMPT = """You are a professional digital forensics analyst specializing in deepfake detection.
You are given structured evidence from an automated multimodal AI deepfake detection system.
Your task is to write a clear, accurate, and technically grounded forensic narrative.

Rules:
1. Be precise and technical but comprehensible to a non-expert reader.
2. Reference specific evidence signals by name.
3. Acknowledge uncertainty where it exists (low confidence, OOD inputs, etc.).
4. NEVER overstate certainty. Verdicts are probabilistic.
5. End with a standard disclaimer.
6. Keep the narrative to 3-5 paragraphs.
7. Do not invent evidence not present in the structured data.
"""


def _build_evidence_summary(evidence: List[Dict]) -> str:
    lines = []
    for e in evidence:
        if e.get("status") != "used":
            continue
        logit = e.get("logit")
        rel = e.get("reliability", 0)
        name = e.get("name", "?")
        direction = "FAKE" if (logit or 0) > 0 else "REAL"
        lines.append(
            f"  - {name}: logit={logit:+.2f} ({direction}), reliability={rel:.2f}"
            + (f", heuristic={e['heuristic']}" if e.get("heuristic") else "")
        )
    return "\n".join(lines) if lines else "  No active evidence."


def _build_prompt(result: Dict[str, Any]) -> str:
    verdict = result.get("verdict", "Inconclusive")
    p_fake = result.get("p_fake", 0.5)
    media = result.get("media_type", "image")
    calibrated = result.get("calibrated", False)
    ood = result.get("ood", False)
    evidence = result.get("evidence", [])
    quality = result.get("quality", {})
    attribution = result.get("attribution", {})
    stability = result.get("stability", {})
    warnings = result.get("warnings", [])
    filename = result.get("filename", "unknown")

    ev_summary = _build_evidence_summary(evidence)
    attr_text = ""
    if attribution and attribution.get("family"):
        attr_text = (
            f"Generator attribution: {attribution['family']} "
            f"(probability {attribution.get('probability', 0):.2f}) — "
            f"{attribution.get('description', '')}"
        )

    stab_text = ""
    if stability:
        score = stability.get("stability_score")
        if score is not None:
            stab_text = f"Verdict stability: {score*100:.0f}% of degraded copies agree."

    warn_text = ""
    if warnings:
        warn_text = "Warnings: " + "; ".join(warnings)

    prompt = f"""
FORENSIC ANALYSIS REQUEST

File: {filename}
Media type: {media}
Verdict: {verdict}
P(manipulated): {p_fake*100:.1f}%
Calibrated: {'Yes' if calibrated else 'No (uncalibrated; treat probabilities with caution)'}
Out-of-distribution flag: {'Yes — input differs from calibration data' if ood else 'No'}

Active Evidence:
{ev_summary}

Quality: {quality}
{attr_text}
{stab_text}
{warn_text}

Write a professional forensic narrative for this result.
""".strip()
    return prompt


def generate_llm_narrative(
    result: Dict[str, Any],
    provider: str = "none",
    model: str = "gpt-4o-mini",
    openai_key: str = "",
    gemini_key: str = "",
) -> Optional[str]:
    """Generate an LLM forensic narrative.

    Returns the narrative string or None if unavailable (caller falls back
    to rule-based narrative).
    """
    if provider == "none" or not provider:
        return None

    prompt = _build_prompt(result)

    if provider == "openai":
        return _call_openai(prompt, model, openai_key)
    elif provider == "gemini":
        return _call_gemini(prompt, gemini_key)
    else:
        log.warning("Unknown LLM provider: %s", provider)
        return None


def _call_openai(prompt: str, model: str, api_key: str) -> Optional[str]:
    if not api_key:
        log.warning("LLM narrative: OpenAI API key not set.")
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        log.info("LLM narrative generated via OpenAI (%s).", model)
        return text
    except ImportError:
        log.warning("openai package not installed. pip install openai")
        return None
    except Exception as e:
        log.warning("OpenAI narrative generation failed: %s", e)
        return None


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    if not api_key:
        log.warning("LLM narrative: Gemini API key not set.")
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        response = model.generate_content(full_prompt)
        text = response.text.strip()
        log.info("LLM narrative generated via Gemini.")
        return text
    except ImportError:
        log.warning("google-generativeai package not installed. pip install google-generativeai")
        return None
    except Exception as e:
        log.warning("Gemini narrative generation failed: %s", e)
        return None
