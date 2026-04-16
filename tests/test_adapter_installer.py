"""Fork-local: install_adapter_evaluator resolves ADAPTER_EVALUATOR_FACTORY
into a callable and installs it under evolve_config['adapter_evaluator']."""
import pytest

from codeevolve.evolution import install_adapter_evaluator


def _fake_factory(evolve_config, args):
    calls = {"count": 0}
    def evaluator(child_sol, timeout_s):
        calls["count"] += 1
        return (0, None, "", "", {"fitness": 1.0})
    evaluator._calls = calls
    return evaluator


def _not_a_factory(evolve_config, args):
    return "not a callable"


def test_install_noop_when_no_factory():
    cfg = {}
    install_adapter_evaluator(cfg, args={})
    assert "adapter_evaluator" not in cfg


def test_install_resolves_and_installs_callable():
    cfg = {
        "ADAPTER_EVALUATOR_FACTORY":
            "tests.test_adapter_installer:_fake_factory",
    }
    install_adapter_evaluator(cfg, args={"inpt_dir": "/tmp"})
    assert callable(cfg["adapter_evaluator"])
    rc, _, warn, err, metrics = cfg["adapter_evaluator"](child_sol=None, timeout_s=1)
    assert rc == 0
    assert metrics["fitness"] == 1.0


def test_install_rejects_non_callable_result():
    cfg = {
        "ADAPTER_EVALUATOR_FACTORY":
            "tests.test_adapter_installer:_not_a_factory",
    }
    with pytest.raises((ImportError, AttributeError, TypeError)):
        install_adapter_evaluator(cfg, args={})


def test_install_rejects_malformed_entry():
    cfg = {"ADAPTER_EVALUATOR_FACTORY": "no_colon_here"}
    with pytest.raises(ValueError):
        install_adapter_evaluator(cfg, args={})
