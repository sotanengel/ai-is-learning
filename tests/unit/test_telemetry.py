"""Tests for OpenTelemetry setup."""

from __future__ import annotations

from opentelemetry.trace import Tracer

from kolb_loop.telemetry import get_tracer, setup_telemetry


def test_setup_telemetry_returns_tracer() -> None:
    tracer = setup_telemetry(service_name="test-svc", export=False)
    assert isinstance(tracer, Tracer)


def test_get_tracer_returns_tracer() -> None:
    setup_telemetry(service_name="test-svc", export=False)
    tracer = get_tracer("kolb_loop.test")
    assert isinstance(tracer, Tracer)


def test_tracer_can_create_span() -> None:
    setup_telemetry(service_name="test-svc", export=False)
    tracer = get_tracer("kolb_loop.test")
    with tracer.start_as_current_span("test-span") as span:
        assert span is not None
        span.set_attribute("test.key", "value")
