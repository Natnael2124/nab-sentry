"""Property 43: Range header model (parser part).

``parse_range`` is compared against an independent, regex-free reference model
on well-formed, near-miss malformed, and multi-range headers. The HTTP
header/body half of Property 43 lives in ``tests/test_api_media_range.py``.

**Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.9, 11.10**
"""

from __future__ import annotations

from hypothesis import example, given, settings
from hypothesis import strategies as st

from nab_sentry.api.media import FullBody, PartialBody, Unsatisfiable, parse_range

ASCII_DIGITS = "0123456789"
MAX_SIZE = 10**6


# ---------------------------------------------------------------------------
# Reference model (character-level, no regex)
# ---------------------------------------------------------------------------


def _is_ascii_digits(s: str) -> bool:
    return all(c in ASCII_DIGITS for c in s)


def reference_range(header, size):
    """Independent model of the design's Range rules table."""
    if header is None:
        return FullBody()
    prefix = "bytes="
    if not header.startswith(prefix):
        return FullBody()
    spec = header[len(prefix):]
    if "," in spec:
        return FullBody()  # multi-range is ignored
    parts = spec.split("-")
    if len(parts) != 2:
        return FullBody()
    first, last = parts
    if not (_is_ascii_digits(first) and _is_ascii_digits(last)):
        return FullBody()
    if first == "" and last == "":
        return FullBody()  # "bytes=-"
    if first == "":
        n = int(last)
        if n == 0 or size == 0:
            return Unsatisfiable()
        return PartialBody(max(0, size - n), size - 1)
    a = int(first)
    if last != "" and a > int(last):
        return FullBody()
    if a >= size:
        return Unsatisfiable()
    end = size - 1 if last == "" else min(int(last), size - 1)
    return PartialBody(a, end)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

sizes = st.one_of(
    st.integers(min_value=0, max_value=MAX_SIZE),
    st.sampled_from([0, 1, 2, MAX_SIZE]),
)

# Numbers biased toward small values, values near typical sizes, and huge values.
numbers = st.one_of(
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=0, max_value=MAX_SIZE + 10),
    st.integers(min_value=10**6, max_value=10**40),
)


@st.composite
def num_text(draw):
    """Decimal text for a number, optionally with leading zeros."""
    n = draw(numbers)
    zeros = draw(st.integers(min_value=0, max_value=3))
    return "0" * zeros + str(n)


@st.composite
def well_formed(draw):
    kind = draw(st.sampled_from(["a-b", "a-", "-n", "-"]))
    if kind == "a-b":
        return f"bytes={draw(num_text())}-{draw(num_text())}"
    if kind == "a-":
        return f"bytes={draw(num_text())}-"
    if kind == "-n":
        return f"bytes=-{draw(num_text())}"
    return "bytes=-"


_unicode_digits = st.sampled_from(["\u0661", "\u06f5", "\u0967", "\uff13", "\u00b2", "\u0e52"])


@st.composite
def malformed(draw):
    base = draw(well_formed())
    mutation = draw(
        st.sampled_from(
            [
                "space_inside",
                "leading_space",
                "trailing_space",
                "trailing_newline",
                "other_unit",
                "missing_prefix",
                "upper_unit",
                "extra_dash",
                "negative",
                "unicode_digit",
                "plus_sign",
                "random_text",
            ]
        )
    )
    if mutation == "space_inside":
        i = draw(st.integers(min_value=0, max_value=len(base)))
        return base[:i] + " " + base[i:]
    if mutation == "leading_space":
        return " " + base
    if mutation == "trailing_space":
        return base + " "
    if mutation == "trailing_newline":
        return base + "\n"
    if mutation == "other_unit":
        unit = draw(st.sampled_from(["items=", "bits=", "byte=", "chars="]))
        return unit + base[len("bytes="):]
    if mutation == "missing_prefix":
        return base[len("bytes="):]
    if mutation == "upper_unit":
        return "Bytes=" + base[len("bytes="):]
    if mutation == "extra_dash":
        return f"{base}-{draw(num_text())}"
    if mutation == "negative":
        return f"bytes=-{draw(num_text())}-{draw(num_text())}"
    if mutation == "unicode_digit":
        body = base[len("bytes="):]
        i = draw(st.integers(min_value=0, max_value=len(body)))
        return "bytes=" + body[:i] + draw(_unicode_digits) + body[i:]
    if mutation == "plus_sign":
        return f"bytes=+{draw(num_text())}-"
    return draw(st.text(max_size=20))


@st.composite
def multi_range(draw):
    parts = draw(st.lists(well_formed(), min_size=2, max_size=4))
    return "bytes=" + ",".join(p[len("bytes="):] for p in parts)


headers = st.one_of(st.none(), well_formed(), malformed(), multi_range())


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _check_invariants(result, size):
    assert isinstance(result, (FullBody, PartialBody, Unsatisfiable))
    if isinstance(result, PartialBody):
        assert size > 0
        assert 0 <= result.start <= result.end <= size - 1


@settings(max_examples=500)
@given(header=headers, size=sizes)
@example(header=None, size=0)
@example(header="bytes=-", size=0)
@example(header="bytes=0-0", size=0)
@example(header="bytes=-0", size=10)
@example(header="bytes=-000", size=10)
@example(header="bytes=5-3", size=10)
@example(header="bytes=5-3", size=0)
@example(header="bytes=10-", size=10)
@example(header="bytes=9-", size=10)
@example(header="bytes=-20", size=10)
@example(header="bytes=0-5\n", size=10)
@example(header="bytes=\u0661-5", size=10)
@example(header="bytes=0-1,3-4", size=10)
@example(header="bytes=1-2-3", size=10)
@example(header="bytes=0-99999999999999999999999", size=10)
def test_parse_range_matches_reference(header, size):
    """Feature: nab-sentry, Property 43: Range header model (parser part).

    **Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.9, 11.10**
    """
    result = parse_range(header, size)
    assert result == reference_range(header, size)
    _check_invariants(result, size)


@settings(max_examples=200)
@given(header=st.one_of(st.none(), malformed(), multi_range()), size=sizes)
def test_ignored_headers_give_full_body(header, size):
    """Malformed, multi-range, and absent headers are ignored (11.1, 11.10).

    **Validates: Requirements 11.1, 11.10**
    """
    if header is not None and reference_range(header, size) != FullBody():
        # A mutation can occasionally still be well-formed (e.g. random text);
        # only assert on genuinely ignored headers.
        return
    assert parse_range(header, size) == FullBody()


def test_unit_examples():
    assert parse_range(None, 100) == FullBody()
    assert parse_range("bytes=0-0", 100) == PartialBody(0, 0)
    assert parse_range("bytes=10-19", 100) == PartialBody(10, 19)
    assert parse_range("bytes=90-200", 100) == PartialBody(90, 99)
    assert parse_range("bytes=50-", 100) == PartialBody(50, 99)
    assert parse_range("bytes=-10", 100) == PartialBody(90, 99)
    assert parse_range("bytes=-500", 100) == PartialBody(0, 99)
    assert parse_range("bytes=100-", 100) == Unsatisfiable()
    assert parse_range("bytes=100-200", 100) == Unsatisfiable()
    assert parse_range("bytes=-0", 100) == Unsatisfiable()
    assert parse_range("bytes=-", 100) == FullBody()
    assert parse_range("bytes=20-10", 100) == FullBody()
    assert parse_range("bytes=0-1,5-6", 100) == FullBody()
    assert parse_range("bytes=0-5\n", 100) == FullBody()
    assert parse_range("bytes=-5", 0) == Unsatisfiable()
    assert parse_range("bytes=0-", 0) == Unsatisfiable()
    assert parse_range("garbage", 0) == FullBody()
