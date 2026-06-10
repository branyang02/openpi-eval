from __future__ import annotations

import importlib.util
from pathlib import Path

PROBE_PATH = Path(__file__).resolve().parents[1] / "inspect_wan_kv_feasibility.py"
PROBE_SPEC = importlib.util.spec_from_file_location("inspect_wan_kv_feasibility", PROBE_PATH)
assert PROBE_SPEC is not None
assert PROBE_SPEC.loader is not None
probe = importlib.util.module_from_spec(PROBE_SPEC)
PROBE_SPEC.loader.exec_module(probe)

WAN_ATTENTION_SOURCE = """
class SelfAttention:
    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention:
    def forward(self, x, y):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)
"""


def test_attention_source_probe_detects_internal_qkv_without_returned_kv() -> None:
    modules = probe.inspect_attention_source(
        WAN_ATTENTION_SOURCE,
        source_path="/tmp/DiffSynth-Studio/diffsynth/models/wan_video_dit.py",
        class_names=("SelfAttention", "CrossAttention"),
    )

    assert [module["class_name"] for module in modules] == ["SelfAttention", "CrossAttention"]
    for module in modules:
        assert module["qkv"]["computed_in_forward"] is True
        assert module["forward"]["returns_key_value_tensors"] is False
        assert module["forward"]["accepts_key_value_cache"] is False
        assert module["forward"]["has_return_key_value_flag"] is False

    self_cacheability = modules[0]["projection_cacheability"]
    cross_cacheability = modules[1]["projection_cacheability"]
    assert self_cacheability["cacheable_projection_targets"] == ("k", "v")
    assert self_cacheability["dynamic_cacheable_projection_targets"] == ("k", "v")
    assert self_cacheability["static_cacheable_projection_targets"] == ()
    assert cross_cacheability["cacheable_projection_targets"] == ("k", "v")
    assert cross_cacheability["static_cacheable_projection_targets"] == ("k", "v")
    assert cross_cacheability["dynamic_cacheable_projection_targets"] == ()


def test_attention_source_probe_detects_public_kv_return() -> None:
    source = """
class CachedAttention:
    def forward(self, x, return_kv=False):
        k = self.k(x)
        v = self.v(x)
        out = self.attn(x, k, v)
        if return_kv:
            return out, (k, v)
        return out
"""

    modules = probe.inspect_attention_source(source, source_path="cached.py", class_names=("CachedAttention",))

    assert modules[0]["forward"]["has_return_key_value_flag"] is True
    assert modules[0]["forward"]["returns_key_value_tensors"] is True


def test_build_report_recommends_hidden_prefix_for_current_wan_source(tmp_path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    wan_source_path = repo_dir / probe.WAN_DIT_RELATIVE_SOURCE
    wan_source_path.parent.mkdir(parents=True)
    wan_source_path.write_text(WAN_ATTENTION_SOURCE)

    script_dir = tmp_path / "world_model_env"
    prefix_source = script_dir / probe.CURRENT_PREFIX_RELATIVE_SOURCE
    prefix_source.parent.mkdir(parents=True)
    prefix_source.write_text(
        """
class WanDiTHiddenFeatureExtractor:
    def _register_hooks(self, captures):
        return [block.register_forward_hook(lambda module, inputs, output: captures.update({0: output}))]

    def extract(self):
        prefix_tokens = "hidden"
        return prefix_tokens


class FrozenDiffSynthWanDiTCurrentPrefixEncoder:
    pass
"""
    )

    report = probe.build_probe_report(diffsynth_repo_dir=repo_dir, script_dir=script_dir)

    assert report["probe"]["loads_model_weights"] is False
    assert report["probe"]["wan_attention_classes"]["available"] is True
    assert report["probe"]["wan_attention_classes"]["inspected"] == ("SelfAttention", "CrossAttention")
    assert report["probe"]["wan_attention_classes"]["missing"] == ()
    assert report["tensor_feasibility"]["forward_exposes_or_returns_wan_key_value_tensors"] is False
    assert report["tensor_feasibility"]["attention_projection_cacheability"]["SelfAttention"][
        "dynamic_cacheable_projection_targets"
    ] == ("k", "v")
    assert report["tensor_feasibility"]["attention_projection_cacheability"]["CrossAttention"][
        "static_cacheable_projection_targets"
    ] == ("k", "v")
    assert report["current_repo_capture"]["capture_mode"] == "hidden_state_prefix_tokens"
    assert report["current_repo_capture"]["captures_key_value_tensors"] is False
    assert report["conclusion"]["recommended_current_mode"] == "hidden_prefix"
    assert "cache-aware SelfAttention/CrossAttention wrappers" in report["conclusion"]["recommendation"]


def test_build_report_marks_wan_source_and_classes_unavailable_when_absent(tmp_path) -> None:
    report = probe.build_probe_report(diffsynth_repo_dir=tmp_path / "missing", script_dir=tmp_path / "world_model_env")

    assert report["probe"]["wan_dit_source"]["available"] is False
    assert report["probe"]["wan_attention_classes"]["available"] is False
    assert report["probe"]["wan_attention_classes"]["inspected"] == ()
    assert report["probe"]["wan_attention_classes"]["missing"] == ("SelfAttention", "CrossAttention")
    assert report["attention_modules"] == []
    assert report["tensor_feasibility"]["forward_exposes_or_returns_wan_key_value_tensors"] is False


def test_resolve_wan_dit_source_from_repo_dir(tmp_path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    wan_source_path = repo_dir / probe.WAN_DIT_RELATIVE_SOURCE
    wan_source_path.parent.mkdir(parents=True)
    wan_source_path.write_text(WAN_ATTENTION_SOURCE)

    assert probe.resolve_wan_dit_source(diffsynth_repo_dir=repo_dir) == wan_source_path.resolve()
