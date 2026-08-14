"""Answer extraction + equivalence. Shared by training and eval.

Twist #4 ("verifier fragility") lives here: the same completions get scored under
several graders, so we can show whether the ranking of GRPO variants depends on
which grader you happened to pick. If it does, that's a methodological result.

Run `python extractors.py` for the self-check.
"""

import re

# Matches 1234 / -1,234 / 12.5 / -1,234.56
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _clean(s: str) -> str:
    return s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")


# ------------------------------------------------------------------------------
# Extractors: completion text -> answer string ("" if none found)
# ------------------------------------------------------------------------------
def strict(text: str) -> str:
    """Requires the '#### <answer>' contract we asked for in the system prompt.
    Harshest grader: a correct answer in the wrong format scores zero."""
    m = re.search(rf"####\s*({_NUM})", text)
    return _clean(m.group(1)) if m else ""


def last_number(text: str) -> str:
    """Prefer '#### x', else fall back to the last number anywhere in the text.
    This is what the original baseline used. It is LENIENT and reward-hackable:
    a model that answers early then rambles still gets credit, and padding the
    completion with numbers can get lucky. Kept precisely so we can measure that."""
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return _clean(m.group(1))
    nums = re.findall(_NUM, text)
    return _clean(nums[-1]) if nums else ""


def _last_boxed(text: str) -> str:
    """Contents of the final \\boxed{...}, with brace matching (nested braces are
    common in LaTeX answers, so a regex would truncate)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    i = text.find("{", idx)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return ""


def boxed(text: str) -> str:
    """\\boxed{...} first, then the '####' contract. Needed for MATH-500, whose
    answers are often non-numeric (fractions, surds, intervals)."""
    b = _last_boxed(text)
    if b:
        return _clean(b)
    return strict(text)


EXTRACTORS = {
    "strict": strict,
    "last_number": last_number,
    "boxed": boxed,
}


# ------------------------------------------------------------------------------
# Equivalence: is the extracted answer the same as gold?
# ------------------------------------------------------------------------------
def _as_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _sympy_equal(pred: str, gold: str) -> bool:
    # ponytail: sympy is optional. Absent -> we simply lose the symbolic grader,
    # which only matters for MATH-500. GSM8K answers are all integers.
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False
    for parser in (parse_expr, parse_latex):
        try:
            a, b = parser(pred), parser(gold)
            if a is not None and b is not None and simplify(a - b) == 0:
                return True
        except Exception:
            continue
    return False


def equal(pred: str, gold: str, mode: str = "numeric") -> bool:
    """mode: 'exact' (string), 'numeric' (float tolerance), 'symbolic' (sympy)."""
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    if mode == "exact":
        return False
    p, g = _as_float(pred), _as_float(gold)
    if p is not None and g is not None:
        return abs(p - g) < 1e-6
    if mode == "symbolic":
        return _sympy_equal(pred, gold)
    return False


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    m = re.search(rf"####\s*({_NUM})", answer_field)
    return _clean(m.group(1)) if m else ""


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    assert strict("blah\n#### 42") == "42"
    assert strict("the answer is 42") == ""            # strict really is strict
    assert last_number("the answer is 42") == "42"     # lenient fallback fires
    assert last_number("blah\n#### 42\nthen 99") == "42"  # '####' wins over later nums
    assert strict("#### -1,234") == "-1234"
    assert strict("#### 12.50") == "12.50"
    assert last_number("no numbers here") == ""
    assert boxed(r"so \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert boxed(r"\boxed{x^{2}}") == "x^{2}"           # nested braces survive
    assert boxed("#### 7") == "7"                       # falls back to strict
    assert equal("12.50", "12.5")                       # numeric tolerance
    assert not equal("12.50", "12.5", mode="exact")
    assert not equal("", "5")
    assert gsm8k_gold("reasoning...\n#### 1,000") == "1000"
    print("extractors: all checks passed")
