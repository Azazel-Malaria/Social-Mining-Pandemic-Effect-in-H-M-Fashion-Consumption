from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from util.io_utils import ensure_dir, save_json

AXIS_SPECS = [
    {
        "axis_id": "formality_structure",
        "positive_endpoint": "formal_structured",
        "negative_endpoint": "relaxed_informal",
        "positive_definition": "formal, polished, structured, professional-looking clothing",
        "negative_definition": "relaxed, informal, casual, easygoing clothing",
        "positive_prompts": [
            "a clothing item that looks formal, polished, and suitable for professional settings",
            "a clothing item that has a structured and refined silhouette",
            "a clothing item that appears elegant, composed, and occasion-ready",
            "a clothing item that would fit a business, ceremony, or formal public setting",
        ],
        "negative_prompts": [
            "a clothing item that looks relaxed, informal, and suitable for casual daily wear",
            "a clothing item that has an easygoing and unstructured appearance",
            "a clothing item that feels laid-back, simple, and non-ceremonial",
            "a clothing item that would fit leisure, informal errands, or relaxed everyday use",
        ],
    },
    {
        "axis_id": "public_private_occasion",
        "positive_endpoint": "public_social_outward",
        "negative_endpoint": "private_home_lounge",
        "positive_definition": "clothing oriented toward public appearance, social activities, or outside-facing occasions",
        "negative_definition": "clothing oriented toward private, home, indoor, or lounge settings",
        "positive_prompts": [
            "a clothing item that looks suitable for being seen in public or social settings",
            "a clothing item that appears outside-facing, presentable, and socially visible",
            "a clothing item that would fit outings, gatherings, or public-facing occasions",
            "a clothing item that emphasizes appearance in social or external environments",
        ],
        "negative_prompts": [
            "a clothing item that looks suitable for staying at home or relaxing indoors",
            "a clothing item that appears private, lounge-oriented, and low-pressure",
            "a clothing item that would fit home, rest, or quiet indoor routines",
            "a clothing item that emphasizes comfort in private rather than public visibility",
        ],
    },
    {
        "axis_id": "comfort_ease",
        "positive_endpoint": "comfortable_easy",
        "negative_endpoint": "restrictive_appearance_first",
        "positive_definition": "comfortable, soft, easy-to-wear clothing",
        "negative_definition": "restrictive, stiff, appearance-first clothing",
        "positive_prompts": [
            "a clothing item that looks comfortable and easy to wear for long periods",
            "a clothing item that appears soft, relaxed, and gentle on the body",
            "a clothing item that prioritizes ease of movement and everyday comfort",
            "a clothing item that feels wearable, practical, and physically forgiving",
        ],
        "negative_prompts": [
            "a clothing item that looks restrictive, stiff, or physically demanding to wear",
            "a clothing item that appears designed more for appearance than bodily comfort",
            "a clothing item that has a constrained, rigid, or difficult-to-move-in look",
            "a clothing item that feels occasion-specific and less suited for relaxed wear",
        ],
    },
    {
        "axis_id": "practical_value",
        "positive_endpoint": "practical_durable_versatile",
        "negative_endpoint": "ornamental_premium_special_use",
        "positive_definition": "practical, durable, versatile, value-oriented clothing",
        "negative_definition": "ornamental, premium, decorative, or special-use clothing",
        "positive_prompts": [
            "a clothing item that looks practical, versatile, and useful for repeated wear",
            "a clothing item that appears durable, functional, and good for everyday value",
            "a clothing item that emphasizes utility, longevity, and easy matching",
            "a clothing item that feels sensible, affordable-looking, and broadly wearable",
        ],
        "negative_prompts": [
            "a clothing item that looks ornamental, premium, and mainly decorative",
            "a clothing item that appears special-use, delicate, or less practical for repeated wear",
            "a clothing item that emphasizes luxury impression over everyday utility",
            "a clothing item that feels more expressive or decorative than durable and versatile",
        ],
    },
    {
        "axis_id": "trend_expressiveness",
        "positive_endpoint": "trendy_expressive_statement",
        "negative_endpoint": "basic_timeless_understated",
        "positive_definition": "fashion-forward, expressive, statement-making clothing",
        "negative_definition": "basic, timeless, understated, low-expression clothing",
        "positive_prompts": [
            "a clothing item that looks trendy, expressive, and fashion-forward",
            "a clothing item that makes a visible style statement",
            "a clothing item that appears distinctive, current, and attention-catching",
            "a clothing item that emphasizes personality, novelty, or fashion expression",
        ],
        "negative_prompts": [
            "a clothing item that looks basic, timeless, and understated",
            "a clothing item that appears simple, plain, and low-key",
            "a clothing item that avoids strong fashion statements or novelty",
            "a clothing item that emphasizes lasting everyday style rather than trend expression",
        ],
    },
    {
        "axis_id": "material_softness",
        "positive_endpoint": "soft_smooth_skin_friendly",
        "negative_endpoint": "stiff_rough_crisp",
        "positive_definition": "soft, smooth, skin-friendly material impression",
        "negative_definition": "stiff, rough, crisp, structured material impression",
        "positive_prompts": [
            "a clothing item that appears soft, smooth, and pleasant against the skin",
            "a clothing item that looks gentle, pliable, and comfortable in texture",
            "a clothing item that suggests a cozy or skin-friendly fabric feel",
            "a clothing item that appears flexible and soft rather than rigid",
        ],
        "negative_prompts": [
            "a clothing item that appears stiff, crisp, or rigid in texture",
            "a clothing item that looks rough, structured, or less soft against the skin",
            "a clothing item that suggests a firm or coarse fabric feel",
            "a clothing item that appears shaped by a hard or inflexible material",
        ],
    },
    {
        "axis_id": "material_thermal_weight",
        "positive_endpoint": "warm_thick_insulating",
        "negative_endpoint": "light_cool_breathable",
        "positive_definition": "warm, thick, insulating, heavy material impression",
        "negative_definition": "light, cool, breathable, airy material impression",
        "positive_prompts": [
            "a clothing item that looks warm, thick, and insulating",
            "a clothing item that appears suitable for cold weather or thermal protection",
            "a clothing item that suggests a dense, cozy, or heavyweight fabric",
            "a clothing item that feels protective against cool temperatures",
        ],
        "negative_prompts": [
            "a clothing item that looks light, cool, and breathable",
            "a clothing item that appears airy, thin, or suitable for warm weather",
            "a clothing item that suggests a lightweight and ventilated fabric",
            "a clothing item that feels cooling rather than insulating",
        ],
    },
    {
        "axis_id": "material_durability",
        "positive_endpoint": "durable_hard_wearing",
        "negative_endpoint": "delicate_fragile",
        "positive_definition": "durable, hard-wearing, robust material impression",
        "negative_definition": "delicate, fragile, easily damaged material impression",
        "positive_prompts": [
            "a clothing item that looks durable, sturdy, and hard-wearing",
            "a clothing item that appears robust enough for repeated everyday use",
            "a clothing item that suggests a resilient and long-lasting material",
            "a clothing item that feels practical and resistant to wear",
        ],
        "negative_prompts": [
            "a clothing item that looks delicate, fragile, or easily damaged",
            "a clothing item that appears to require careful handling",
            "a clothing item that suggests a fine, fragile, or less durable material",
            "a clothing item that feels decorative or delicate rather than hard-wearing",
        ],
    },
    {
        "axis_id": "color_temperature",
        "positive_endpoint": "warm_toned",
        "negative_endpoint": "cool_toned",
        "positive_definition": "warm color tones such as red, orange, yellow, beige, or warm brown",
        "negative_definition": "cool color tones such as blue, green, grey, teal, or cool purple",
        "positive_prompts": [
            "a clothing item dominated by warm color tones such as red, orange, beige, or warm brown",
            "a clothing item with a warm, cozy, or sunlit color impression",
            "a clothing item whose color palette feels warm-toned rather than cool-toned",
            "a clothing item with colors associated with warmth, earthiness, or golden tones",
        ],
        "negative_prompts": [
            "a clothing item dominated by cool color tones such as blue, green, grey, or teal",
            "a clothing item with a cool, calm, or blue-toned color impression",
            "a clothing item whose color palette feels cool-toned rather than warm-toned",
            "a clothing item with colors associated with coolness, water, shade, or steel tones",
        ],
    },
    {
        "axis_id": "color_lightness",
        "positive_endpoint": "light_pale_bright",
        "negative_endpoint": "dark_deep",
        "positive_definition": "light, pale, bright color appearance",
        "negative_definition": "dark, deep, low-lightness color appearance",
        "positive_prompts": [
            "a clothing item dominated by light, pale, or bright colors",
            "a clothing item with a high-lightness color appearance",
            "a clothing item that appears visually light rather than dark",
            "a clothing item with colors closer to white, pastel, or pale tones",
        ],
        "negative_prompts": [
            "a clothing item dominated by dark, deep, or low-lightness colors",
            "a clothing item with a dark color appearance",
            "a clothing item that appears visually deep rather than light",
            "a clothing item with colors closer to black, navy, dark brown, or deep tones",
        ],
    },
    {
        "axis_id": "color_chroma",
        "positive_endpoint": "vivid_saturated_colorful",
        "negative_endpoint": "muted_neutral_desaturated",
        "positive_definition": "vivid, saturated, colorful appearance",
        "negative_definition": "muted, neutral, desaturated appearance",
        "positive_prompts": [
            "a clothing item with vivid, saturated, and colorful tones",
            "a clothing item that looks bright in color intensity and visually lively",
            "a clothing item with bold color presence rather than muted tones",
            "a clothing item whose colors feel expressive, saturated, and eye-catching",
        ],
        "negative_prompts": [
            "a clothing item with muted, neutral, or desaturated tones",
            "a clothing item that looks subdued in color intensity",
            "a clothing item with quiet color presence rather than vivid tones",
            "a clothing item whose colors feel understated, washed, or neutral",
        ],
    },
    {
        "axis_id": "design_complexity",
        "positive_endpoint": "detailed_embellished_patterned",
        "negative_endpoint": "minimal_plain_simple",
        "positive_definition": "detailed, embellished, patterned, visually complex design",
        "negative_definition": "minimal, plain, simple, visually clean design",
        "positive_prompts": [
            "a clothing item with visible design details, decoration, or embellishment",
            "a clothing item that looks visually complex or richly designed",
            "a clothing item with patterns, layers, trims, or distinctive design elements",
            "a clothing item that draws attention through decorative or complex styling",
        ],
        "negative_prompts": [
            "a clothing item with a minimal, plain, and simple design",
            "a clothing item that looks visually clean and undecorated",
            "a clothing item with little pattern, embellishment, or extra detail",
            "a clothing item that emphasizes simplicity rather than design complexity",
        ],
    },
    {
        "axis_id": "fit_looseness",
        "positive_endpoint": "loose_relaxed_draped",
        "negative_endpoint": "tight_fitted_body_hugging",
        "positive_definition": "loose, relaxed, draped fit",
        "negative_definition": "tight, fitted, body-hugging fit",
        "positive_prompts": [
            "a clothing item with a loose, relaxed, or draped fit",
            "a clothing item that leaves room around the body and looks easy to move in",
            "a clothing item with an oversized or flowing silhouette",
            "a clothing item that appears roomy rather than body-hugging",
        ],
        "negative_prompts": [
            "a clothing item with a tight, fitted, or body-hugging fit",
            "a clothing item that closely follows the shape of the body",
            "a clothing item with a slim, narrow, or contouring silhouette",
            "a clothing item that appears fitted rather than roomy",
        ],
    },
]

AXIS_IDS = [x["axis_id"] for x in AXIS_SPECS]
COMPOSITE_IDS = ["office_like", "homewear_like", "casual_like", "social_outing_like", "value_basic"]


def default_axis_prompt_schema() -> dict:
    return {
        "schema_version": "axis_prompt_v1",
        "scope": "generic_clothing",
        "source": "built_in_default",
        "axes": [dict(x) for x in AXIS_SPECS],
    }


def _dedup_keep(xs: Iterable[Any], n: int = 4) -> list[str]:
    out, seen = [], set()
    for x in xs or []:
        s = re.sub(r"\s+", " ", str(x).strip().strip('"\'`-•.;'))
        if not s:
            continue
        if not s.lower().startswith("a clothing item"):
            s = "a clothing item that " + s[0].lower() + s[1:] if s else s
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= n:
            break
    return out


def _validate_axis(axis_obj: dict, fallback: dict, prompts_per_polarity: int = 4) -> dict:
    ans = dict(fallback)
    if isinstance(axis_obj, dict):
        pos = _dedup_keep(axis_obj.get("positive_prompts", []), prompts_per_polarity)
        neg = _dedup_keep(axis_obj.get("negative_prompts", []), prompts_per_polarity)
        if len(pos) >= 2:
            ans["positive_prompts"] = (pos + fallback["positive_prompts"])[:prompts_per_polarity]
        if len(neg) >= 2:
            ans["negative_prompts"] = (neg + fallback["negative_prompts"])[:prompts_per_polarity]
    ans["positive_prompts"] = ans["positive_prompts"][:prompts_per_polarity]
    ans["negative_prompts"] = ans["negative_prompts"][:prompts_per_polarity]
    return ans


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _qwen_one_axis(generator: Any, spec: dict, prompts_per_polarity: int = 4) -> tuple[dict, str]:
    system = "You generate fixed semantic-axis prompts for fashion clothing analysis. Return valid JSON only."
    user = f"""
Generate positive and negative prompts for one semantic axis of generic clothing items.

Rules:
1. Prompts must describe visible, textual, material, design, color, fit, or usage attributes of clothing.
2. Do not mention COVID, pandemic, lockdown, quarantine, remote work, or any historical event.
3. Do not use simple negation such as "not formal" or "not comfortable".
4. Positive and negative prompts must be two interpretable endpoints of the same axis.
5. Every prompt must start with or naturally fit the phrase "a clothing item that ...".
6. Return JSON only.
7. Generate exactly {prompts_per_polarity} positive prompts and exactly {prompts_per_polarity} negative prompts.

Axis metadata:
axis_id = {spec['axis_id']}
positive_endpoint = {spec['positive_endpoint']}
negative_endpoint = {spec['negative_endpoint']}
positive_definition = {spec['positive_definition']}
negative_definition = {spec['negative_definition']}

Output JSON schema:
{{
  "axis_id": "{spec['axis_id']}",
  "positive_endpoint": "{spec['positive_endpoint']}",
  "negative_endpoint": "{spec['negative_endpoint']}",
  "positive_prompts": ["...", "...", "...", "..."],
  "negative_prompts": ["...", "...", "...", "..."]
}}
""".strip()
    raw = generator._chat(system, user)
    obj = _extract_json(raw)
    return _validate_axis(obj, spec, prompts_per_polarity), raw


def generate_axis_prompts_with_qwen(
    qwen_model: str,
    cuda_id: int = 0,
    mock: bool = False,
    prompts_per_polarity: int = 4,
    max_input_tokens: int = 4096,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> dict:
    """Generate fixed-template generic clothing axis prompts with Qwen.

    If Qwen is unavailable or a particular axis returns invalid JSON, this function
    falls back to the built-in prompt bank for that axis.  The axis IDs and endpoint
    definitions are never invented by the LLM; Qwen only rewrites endpoint prompts.
    """
    if mock:
        schema = default_axis_prompt_schema()
        schema["source"] = "built_in_mock"
        return schema
    try:
        from util.retrieve_generate_knowledge_prompts import QwenGenerator
        gen = QwenGenerator(
            qwen_model,
            cuda_id=cuda_id,
            mock=False,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        axes = []
        raws = {}
        for spec in AXIS_SPECS:
            try:
                axis, raw = _qwen_one_axis(gen, spec, prompts_per_polarity)
                axis["generation_status"] = "qwen_ok"
                axes.append(axis)
                raws[spec["axis_id"]] = raw
            except Exception as exc:
                fb = dict(spec)
                fb["generation_status"] = f"fallback:{type(exc).__name__}:{exc}"
                axes.append(fb)
        return {
            "schema_version": "axis_prompt_v1",
            "scope": "generic_clothing",
            "source": f"qwen:{qwen_model}",
            "axes": axes,
            "raw_generations": raws,
        }
    except Exception as exc:
        schema = default_axis_prompt_schema()
        schema["source"] = f"built_in_fallback_after_qwen_error:{type(exc).__name__}:{exc}"
        return schema


def load_or_generate_axis_prompt_schema(
    out_json: Path,
    qwen_model: str,
    cuda_id: int,
    force_regenerate: bool = False,
    auto_generate: bool = True,
    mock: bool = False,
    prompts_per_polarity: int = 4,
    max_input_tokens: int = 4096,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> dict:
    out_json = Path(out_json)
    ensure_dir(out_json.parent)
    if out_json.exists() and not force_regenerate:
        with out_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    if auto_generate:
        schema = generate_axis_prompts_with_qwen(
            qwen_model=qwen_model,
            cuda_id=cuda_id,
            mock=mock,
            prompts_per_polarity=prompts_per_polarity,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        schema = default_axis_prompt_schema()
    save_json(schema, out_json)
    return schema


def axis_prompt_meta_from_schema(schema: dict) -> pd.DataFrame:
    rows = []
    for axis in schema.get("axes", []):
        aid = str(axis.get("axis_id", "")).strip()
        if not aid:
            continue
        for polarity, key in [("positive", "positive_prompts"), ("negative", "negative_prompts")]:
            for j, prompt in enumerate(axis.get(key, []) or []):
                rows.append({
                    "axis_id": aid,
                    "polarity": polarity,
                    "prompt_rank": int(j),
                    "prompt_id": f"{aid}:{polarity}:{j}",
                    "prompt_text": str(prompt),
                    "positive_endpoint": axis.get("positive_endpoint", ""),
                    "negative_endpoint": axis.get("negative_endpoint", ""),
                    "positive_definition": axis.get("positive_definition", ""),
                    "negative_definition": axis.get("negative_definition", ""),
                    "scope": schema.get("scope", "generic_clothing"),
                    "source": schema.get("source", "unknown"),
                })
    return pd.DataFrame(rows)


def robust_sigmoid_scores(raw_margin: np.ndarray, tau: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(raw_margin, dtype=np.float64)
    med = np.nanmedian(m)
    mad = np.nanmedian(np.abs(m - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        sd = np.nanstd(m)
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    z = (m - med) / scale
    z = np.clip(z / max(float(tau), 1e-6), -30, 30)
    score = 1.0 / (1.0 + np.exp(-z))
    return score.astype(np.float32), ((m - med) / scale).astype(np.float32)
