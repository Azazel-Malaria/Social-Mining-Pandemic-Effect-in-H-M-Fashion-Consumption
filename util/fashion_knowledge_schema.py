from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

STATIC_DIMS = [
    "material", "color", "design", "occasion", "target", "value",
    "positive", "negative", "social_indicating", "social_reflecting",
]
TEMPORAL_DIMS = ["temporal", "seasonal"]
ALL_DIMS = STATIC_DIMS + TEMPORAL_DIMS

DEFAULT_PROMPTS_PER_DIM: Dict[str, int] = {
    "material": 8,
    "color": 6,
    "design": 10,
    "occasion": 6,
    "target": 5,
    "value": 6,
    "positive": 6,
    "negative": 6,
    "social_indicating": 4,
    "social_reflecting": 4,
    "temporal": 4,
    "seasonal": 4,
}

RELATIONS: Dict[str, str] = {
    "material": "with",
    "color": "with",
    "design": "with",
    "occasion": "used for",
    "target": "suitable for",
    "value": "valued for",
    "positive": "preferred for",
    "negative": "with common concerns about",
    "social_indicating": "indicating",
    "social_reflecting": "reflecting",
    "temporal": "reflecting temporal shift toward",
    "seasonal": "associated with seasonal demand for",
}

DIM_ALIASES = {
    "social": ["social_indicating", "social_reflecting"],
}

QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "material": [
        "cotton fabric breathable comfort customer reviews",
        "linen lightweight summer fabric customer reviews",
        "silk satin smooth texture dressy customer reviews",
        "denim durable casual fabric customer reviews",
        "wool warm formal fabric customer reviews",
        "nylon polyester synthetic quick dry outdoor customer reviews",
        "stretch elastic spandex fit movement customer reviews",
        "fabric thinness shrinkage wrinkling material complaints",
    ],
    "color": [
        "black white beige gray neutral color matching customer reviews",
        "bright printed colors high contrast motifs customer reviews",
        "pastel soft color palette customer reviews",
        "dark color fading washing customer complaints",
        "color accuracy product photo customer reviews",
        "seasonal colors spring summer autumn winter fashion reviews",
    ],
    "design": [
        "minimalist clean design simple tailoring customer reviews",
        "cartoon pattern playful graphic youth design customer reviews",
        "retro vintage y2k design fashion customer reviews",
        "streetwear oversized utility cargo design customer reviews",
        "formal classic office business casual tailoring reviews",
        "romantic lace floral feminine design customer reviews",
        "sporty athleisure functional design customer reviews",
        "postmodern asymmetrical experimental design customer reviews",
        "cute kawaii character print design customer reviews",
        "basic everyday versatile design customer reviews",
    ],
    "occasion": [
        "daily wear home school casual outings customer reviews",
        "office work business casual formal occasions customer reviews",
        "party date social event dressy occasions reviews",
        "sport gym outdoor travel active use customer reviews",
        "loungewear homewear comfort remote work customer reviews",
        "commute city walking practical everyday use reviews",
    ],
    "target": [
        "women men unisex target customer reviews",
        "kids children toddlers youth consumers reviews",
        "teen young adults playful style reviews",
        "mature consumers classic practical clothing reviews",
        "plus size petite fit target group customer reviews",
    ],
    "value": [
        "affordable price value for money customer reviews",
        "durable repeated daily use worth the price reviews",
        "premium quality expensive price customer reviews",
        "cheap low price quality concerns reviews",
        "easy care washing durability value customer reviews",
        "fabric quality price matching expectation reviews",
    ],
    "positive": [
        "positive reviews comfort soft fabric fit customer preference",
        "positive reviews cute design color appearance preference",
        "positive reviews durable quality repeated use preference",
        "positive reviews affordable value easy matching preference",
        "positive reviews size accurate flattering fit preference",
        "positive reviews versatile daily wear preference",
    ],
    "negative": [
        "negative reviews inconsistent sizing fit complaints",
        "negative reviews thin fabric poor quality complaints",
        "negative reviews color fading print fading complaints",
        "negative reviews wrinkling shrinkage washing complaints",
        "negative reviews uncomfortable stiff scratchy material complaints",
        "negative reviews poor stitching limited durability complaints",
    ],
    "social_indicating": [
        "casualization low formality clothing consumption reviews",
        "comfort driven daily consumption homewear lifestyle reviews",
        "youth oriented self expression playful fashion reviews",
        "value sensitive practical consumption affordability reviews",
    ],
    "social_reflecting": [
        "demand for low formality clothing everyday comfort reviews",
        "workwear office return structured dressing reviews",
        "personalized expressive fashion identity reviews",
        "basic durable consumption repeated use reviews",
    ],
    "temporal": [
        "post pandemic comfort home casual demand reviews",
        "pandemic remote work formal clothing demand reviews",
        "temporal shift toward value durability affordability reviews",
        "temporal shift toward outdoor active comfortable clothing reviews",
    ],
    "seasonal": [
        "summer lightweight breathable seasonal demand reviews",
        "winter warm wool knit seasonal demand reviews",
        "spring bright colors floral seasonal demand reviews",
        "autumn layering neutral color seasonal demand reviews",
    ],
}

FALLBACK_ATTRIBUTES: Dict[str, List[str]] = {
    "material": [
        "soft fabric and everyday comfort", "breathable material for repeated daily use",
        "durable textile quality and acceptable thickness", "stretch or elastic fabric supporting movement",
        "lightweight fabric suitable for warm weather", "warm structured material for cooler weather",
        "easy-care fabric for washing and repeated use", "possible concerns about thinness or shrinkage",
    ],
    "color": [
        "neutral colors supporting easy matching", "bright colors and expressive visual appearance",
        "dark colors suitable for versatile daily outfits", "pastel or soft palette for casual styling",
        "high-contrast printed colors", "possible concerns about color fading or mismatch",
    ],
    "design": [
        "minimalist everyday design", "cartoon or graphic patterns and playful expression",
        "formal tailoring and structured appearance", "streetwear or utility-inspired styling",
        "retro or vintage visual language", "sporty functional design",
        "romantic floral or decorative details", "basic versatile styling",
        "postmodern or experimental visual elements", "cute youth-oriented motifs",
    ],
    "occasion": [
        "daily wear, home, school, and casual outings", "office wear and business-casual settings",
        "party, date, and low-frequency social events", "sport, outdoor, travel, and active use",
        "homewear and comfort-oriented indoor use", "commuting and practical city movement",
    ],
    "target": [
        "women, men, or unisex consumers", "kids and youth consumers", "young consumers seeking expressive casual outfits",
        "mature consumers seeking practical and stable styles", "plus-size or fit-sensitive consumers",
    ],
    "value": [
        "affordable price and repeated daily use", "durability and acceptable quality for the price",
        "easy care, washability, and practical use", "premium appearance and higher quality expectations",
        "basic value and versatile matching", "possible mismatch between price and perceived quality",
    ],
    "positive": [
        "soft touch, comfort, and easy matching", "cute or distinctive appearance", "stable fit and daily practicality",
        "durability and repeated usability", "affordability and value for money", "versatility across casual outfits",
    ],
    "negative": [
        "inconsistent sizing and fit uncertainty", "thin fabric and poor material quality",
        "wrinkling, shrinkage, or washing problems", "print or color fading",
        "poor stitching or limited durability", "limited suitability for formal settings",
    ],
    "social_indicating": [
        "casualization and low-formality dressing", "comfort-driven daily consumption",
        "youth-oriented self-expression", "value-sensitive practical consumption",
    ],
    "social_reflecting": [
        "demand for everyday comfort and practical clothing", "shift away from rigid formal dressing",
        "personalized and expressive fashion preference", "basic durable consumption under budget sensitivity",
    ],
    "temporal": [
        "comfort, home use, and casual daily consumption", "reduced demand for formal office-oriented dressing",
        "value-sensitive and durable practical purchases", "active, outdoor, or flexible lifestyle clothing",
    ],
    "seasonal": [
        "lightweight breathable summer use", "warm layered winter use", "bright spring and summer colors",
        "autumn layering and neutral styling",
    ],
}


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def render_prompt(subject: str, dim: str, attribute: str) -> str:
    subject = str(subject).strip() or "fashion item"
    relation = RELATIONS.get(dim, "with")
    attr = str(attribute).strip().strip(".;")
    if not attr:
        attr = "general fashion attributes"
    return f"{subject} {relation} {attr}".strip()


def dimension_output_count(dim: str, custom: str | None = None) -> int:
    if custom:
        pairs = [p.strip() for p in custom.split(",") if p.strip()]
        for p in pairs:
            if ":" in p:
                k, v = p.split(":", 1)
                if k.strip() == dim:
                    return int(v)
    return DEFAULT_PROMPTS_PER_DIM.get(dim, 4)


def get_query_variants(subject: str, dim: str, max_variants: int | None = None) -> List[str]:
    expansions = QUERY_EXPANSIONS.get(dim, [dim])
    out = [f"{subject} {e}" for e in expansions]
    if max_variants is not None and max_variants > 0:
        out = out[:max_variants]
    return out


def fallback_attributes(dim: str, n: int) -> List[str]:
    vals = FALLBACK_ATTRIBUTES.get(dim, ["general fashion attributes"])
    out = []
    while len(out) < n:
        out.extend(vals)
    return out[:n]
