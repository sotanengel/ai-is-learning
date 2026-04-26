"""Tests for configuration loading."""

from pathlib import Path

from kolb_loop.config import Settings, load_settings


def test_load_default_settings() -> None:
    settings = load_settings(Path("/nonexistent/path.yaml"))
    assert isinstance(settings, Settings)
    assert settings.sidecar.ingress.port == 8080
    assert settings.sidecar.fail_open is True


def test_load_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "test.yaml"
    config_file.write_text(
        """
sidecar:
  ingress:
    port: 9090
  fail_open: false
backends:
  main:
    base_url: http://localhost:11434/v1
    model: test-model
  reflector:
    base_url: http://localhost:11434/v1
    model: test-model
  embedder:
    base_url: http://localhost:11434/v1
    model: bge-m3
"""
    )
    settings = load_settings(config_file)
    assert settings.sidecar.ingress.port == 9090
    assert settings.sidecar.fail_open is False
    assert settings.backends.main.model == "test-model"


def test_default_backends_config() -> None:
    settings = load_settings(Path("/nonexistent/path.yaml"))
    assert settings.backends.main.base_url == "http://localhost:11434/v1"
    assert settings.learning.injection.ab_test_ratio == 0.2
    assert settings.learning.injection.max_concepts_per_call == 3
