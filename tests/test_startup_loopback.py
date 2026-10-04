"""Property 52 (ensure_loopback half): only loopback binding is allowed.

The ``serve()`` half (exit before any listening socket is created) is covered by task 14.14.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.startup import LOOPBACK_HOST, StartupError, ensure_loopback

LOOPBACK_MESSAGE = "MVP permits only loopback binding"

# Hosts that look plausible but are not exactly "127.0.0.1".
NEAR_MISSES = [
    "0.0.0.0",
    "::",
    "::1",
    "[::1]",
    "localhost",
    "LOCALHOST",
    "127.0.0.2",
    "127.1",
    "127.000.000.001",
    "0177.0.0.1",
    "2130706433",
    "192.168.1.10",
    "10.0.0.5",
    " 127.0.0.1",
    "127.0.0.1 ",
    "127.0.0.1\n",
    "127.0.0.1:8000",
    "",
]

_ipv4 = st.tuples(*[st.integers(0, 255)] * 4).map(lambda t: ".".join(map(str, t)))
_padded_loopback = st.tuples(
    st.text(alphabet=" \t\r\n\x00", max_size=3), st.text(alphabet=" \t\r\n\x00", max_size=3)
).filter(lambda p: p != ("", "")).map(lambda p: p[0] + LOOPBACK_HOST + p[1])

non_loopback_strings = st.one_of(
    st.sampled_from(NEAR_MISSES),
    _ipv4,
    _padded_loopback,
    st.ip_addresses(v=6).map(str),
    st.text(),
).filter(lambda h: h != LOOPBACK_HOST)

non_string_hosts = st.one_of(
    st.none(),
    st.integers(),
    st.floats(allow_nan=True),
    st.binary(),
    st.just(LOOPBACK_HOST.encode()),
    st.lists(st.just(LOOPBACK_HOST), max_size=2),
    st.tuples(st.just(LOOPBACK_HOST), st.integers(0, 65535)),
)


# Feature: nab-sentry, Property 52: Only loopback binding is allowed
# **Validates: Requirements 10.1, 18.1, 18.5**
@given(host=non_loopback_strings)
def test_non_loopback_string_hosts_are_refused(host: str) -> None:
    with pytest.raises(StartupError, match=LOOPBACK_MESSAGE):
        ensure_loopback(host)


# Feature: nab-sentry, Property 52: Only loopback binding is allowed
# **Validates: Requirements 10.1, 18.1, 18.5**
@given(host=non_string_hosts)
def test_non_string_hosts_are_refused(host: object) -> None:
    with pytest.raises(StartupError, match=LOOPBACK_MESSAGE):
        ensure_loopback(host)


def test_exact_loopback_is_allowed() -> None:
    assert LOOPBACK_HOST == "127.0.0.1"
    assert ensure_loopback("127.0.0.1") is None


@pytest.mark.parametrize("host", NEAR_MISSES)
def test_near_miss_hosts_are_refused(host: str) -> None:
    with pytest.raises(StartupError, match=LOOPBACK_MESSAGE):
        ensure_loopback(host)
