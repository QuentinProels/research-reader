"""Rewrite text into something a speech model says correctly.

Kokoro's grapheme-to-phoneme stage treats any period as a sentence end, so "95.6%"
comes out as "ninety-five" — full stop — "six percent". Measured on rendered audio, a
decimal opened a 0.32s gap mid-number. The same happens for "et al.", "e.g.", "Fig. 3",
"3.2.1", and thousands separators. Symbols fare no better: "±", "×" and Greek letters
are either skipped or mispronounced.

This runs before chunking, not just before synthesis, because app.tts.chunk_text splits
sentences on ". " — so "e.g. attention" was also being cut into two chunks, adding a
second gap on top of Kokoro's.

Every rule here is idempotent: normalising twice is the same as normalising once.
"""

import re

# Abbreviations that can genuinely end a sentence. If one is followed by a capital it
# probably did, so the expansion keeps a period; otherwise it must not, or the sentence
# acquires a stop in the middle of itself.
#
# Only these two qualify. "e.g." and "vs." look sentence-final by the same test but
# almost never are -- "e.g. BERT" and "vs. Transformer" are a following capitalised
# model name, not a new sentence -- so they belong in the group below.
_SENTENCE_AMBIGUOUS = {
    r"et\s+al\.": "and colleagues",
    r"etc\.": "and so on",
}

# These introduce a following term and are never sentence-final, so the period always
# goes, whatever case the next word happens to be.
_REFERENCE_ABBREVIATIONS = {
    r"(?<!\w)e\.g\.": "for example",
    r"(?<!\w)i\.e\.": "that is",
    r"(?<!\w)vs\.": "versus",
    r"(?<!\w)cf\.": "compare",
    r"(?<!\w)w\.r\.t\.": "with respect to",
    r"(?<!\w)approx\.": "approximately",
    r"\bFig\.": "Figure",
    r"\bFigs\.": "Figures",
    r"\bEq\.": "Equation",
    r"\bEqs\.": "Equations",
    r"\bTab\.": "Table",
    r"\bSec\.": "Section",
    r"\bSect\.": "Section",
    r"\bRef\.": "Reference",
    r"\bApp\.": "Appendix",
    r"\bAlg\.": "Algorithm",
}

_SYMBOLS = {
    "%": " percent",
    "±": " plus or minus ",
    "≈": " approximately ",
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
    "≠": " not equal to ",
    "→": " to ",
    "∞": " infinity ",
    "∇": " gradient ",
    "∑": " the sum of ",
    "∏": " the product of ",
    "√": " the square root of ",
    "°": " degrees ",
    "&": " and ",
}

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
    "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi",
    "Ψ": "psi", "Ω": "omega",
}

_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_NUMERIC_RANGE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
_MODEL_SUFFIX = re.compile(r"(?<=[A-Za-z])-(?=\d)")
_DIMENSIONS = re.compile(r"(?<=\d)\s*[×x](?=\s*\d)")
_MULTIPLY = re.compile(r"\s*×\s*")
_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _expand_ambiguous(text: str) -> str:
    for pattern, replacement in _SENTENCE_AMBIGUOUS.items():
        # The abbreviation matches case-insensitively, but the following capital must
        # not: a global re.IGNORECASE makes [A-Z] match lowercase too, which appended a
        # sentence-ending period to every mid-sentence "e.g." it was meant to protect.
        text = re.sub(rf"(?<!\w)(?i:{pattern})(?=\s+[A-Z])", f"{replacement}.", text)
        text = re.sub(rf"(?<!\w)(?i:{pattern})", replacement, text)
    return text


def normalize(text: str) -> str:
    """Make one passage safe to hand to the speech model."""
    if not text:
        return text

    text = _expand_ambiguous(text)
    for pattern, replacement in _REFERENCE_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    for symbol, spoken in _GREEK.items():
        text = text.replace(symbol, spoken)

    # Dimensions read as "224 by 224"; a bare multiplication sign reads as "times".
    text = _DIMENSIONS.sub(" by ", text)
    text = _MULTIPLY.sub(" times ", text)

    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _DECIMAL_POINT.sub(" point ", text)
    text = _NUMERIC_RANGE.sub(" to ", text)
    text = _MODEL_SUFFIX.sub(" ", text)

    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)

    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()
