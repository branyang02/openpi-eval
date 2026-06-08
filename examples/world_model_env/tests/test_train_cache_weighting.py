"""Tests for the opt-in source/cache weighting API of the Wan action expert trainer.

These cover the source-mass ``WeightedRandomSampler`` (per-row weight =
source_weight / source_row_count so the expected source fraction is independent of
cache size), its validation, deterministic seeding, and the metrics/checkpoint
metadata recorded for both the default concat-shuffle path and the weighted path.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler

from train_pi05_wan_action_expert import (
    Args,
    _make_weighted_train_sampler,
    _normalize_cache_weights,
    _resolve_cache_weights,
    _source_mass_row_weights,
    _train_source_indices,
    run_train_eval,
)
from world_model.pi05_wan_action_expert import write_fake_prefix_cache


def _write_cache(cache_dir, *, num_rows: int, seed: int, action_value: float | None = None) -> None:
    write_fake_prefix_cache(
        cache_dir,
        num_rows=num_rows,
        prefix_tokens=2,
        prefix_dim=8,
        state_dim=5,
        action_horizon=4,
        action_dim=3,
        seed=seed,
    )
    if action_value is not None:
        for row_path in sorted(cache_dir.glob("*.pt")):
            row = torch.load(row_path, weights_only=False)
            row["actions"] = torch.full_like(row["actions"], action_value)
            torch.save(row, row_path)


def _train_args(**overrides) -> Args:
    base = {
        "epochs": 0,
        "batch_size": 2,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "sample_steps": 2,
        "device": "cpu",
    }
    base.update(overrides)
    return Args(**base)


def _drawn_sources(sampler: WeightedRandomSampler, source_indices: list[int]) -> list[int]:
    return [source_indices[index] for index in sampler]


# --- validation -------------------------------------------------------------------


def test_cache_weights_length_must_match_train_caches() -> None:
    with pytest.raises(ValueError, match="one weight per train cache"):
        _resolve_cache_weights(cache_weights=(1.0,), samples_per_epoch=None, cache_weight_seed=None, num_caches=2)


def test_cache_weights_reject_negative_weight() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _resolve_cache_weights(cache_weights=(1.0, -0.5), samples_per_epoch=None, cache_weight_seed=None, num_caches=2)


def test_cache_weights_reject_non_finite_weight() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _resolve_cache_weights(
            cache_weights=(1.0, math.inf), samples_per_epoch=None, cache_weight_seed=None, num_caches=2
        )


def test_cache_weights_require_positive_sum() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        _resolve_cache_weights(cache_weights=(0.0, 0.0), samples_per_epoch=None, cache_weight_seed=None, num_caches=2)


def test_samples_per_epoch_requires_cache_weights() -> None:
    with pytest.raises(ValueError, match="require a non-empty cache_weights"):
        _resolve_cache_weights(cache_weights=(), samples_per_epoch=10, cache_weight_seed=None, num_caches=2)


def test_cache_weight_seed_requires_cache_weights() -> None:
    with pytest.raises(ValueError, match="require a non-empty cache_weights"):
        _resolve_cache_weights(cache_weights=(), samples_per_epoch=None, cache_weight_seed=3, num_caches=2)


def test_samples_per_epoch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_cache_weights(cache_weights=(1.0, 1.0), samples_per_epoch=0, cache_weight_seed=None, num_caches=2)


def test_empty_cache_weights_resolve_to_empty_tuples() -> None:
    raw, normalized = _resolve_cache_weights(
        cache_weights=(), samples_per_epoch=None, cache_weight_seed=None, num_caches=2
    )
    assert raw == ()
    assert normalized == ()


def test_resolve_cache_weights_normalizes_to_unit_sum() -> None:
    raw, normalized = _resolve_cache_weights(
        cache_weights=(3.0, 1.0), samples_per_epoch=None, cache_weight_seed=None, num_caches=2
    )
    assert raw == (3.0, 1.0)
    assert normalized == pytest.approx((0.75, 0.25))
    assert sum(normalized) == pytest.approx(1.0)


def test_source_mass_rejects_positive_weight_with_no_rows() -> None:
    # Source 1 carries half the mass but contributes no rows to the train split.
    with pytest.raises(ValueError, match="no rows from that cache"):
        _source_mass_row_weights([0, 0, 0], [0.5, 0.5], num_sources=2)


# --- source mapping ---------------------------------------------------------------


def test_train_source_indices_maps_concat_and_subset() -> None:
    concat = ConcatDataset([[0, 1, 2], [3, 4]])  # cumulative_sizes == [3, 5]

    assert _train_source_indices(concat, concat) == [0, 0, 0, 1, 1]

    subset = Subset(concat, [4, 0, 3])
    assert _train_source_indices(subset, concat) == [1, 0, 1]


# --- sampler behavior -------------------------------------------------------------


def test_equal_weights_draw_balanced_sources_despite_unequal_sizes() -> None:
    # Source 0 has 9 rows, source 1 has 1 row. With equal weights, source-mass
    # semantics must still draw each source ~50% of the time.
    source_indices = [0] * 9 + [1] * 1
    _, normalized = _normalize_cache_weights((1.0, 1.0), num_caches=2)
    generator = torch.Generator().manual_seed(123)

    sampler = _make_weighted_train_sampler(
        source_indices, normalized, num_sources=2, num_samples=8000, generator=generator
    )
    drawn = _drawn_sources(sampler, source_indices)

    frac_source_1 = sum(source == 1 for source in drawn) / len(drawn)
    assert frac_source_1 == pytest.approx(0.5, abs=0.05)


def test_skewed_weights_match_expected_fraction_independent_of_cache_size() -> None:
    # Source 0 is small (2 rows) but heavily weighted; source 1 is large (18 rows)
    # but lightly weighted. Drawn fractions must track the normalized weights, not
    # the cache sizes.
    source_indices = [0] * 2 + [1] * 18
    _, normalized = _normalize_cache_weights((0.8, 0.2), num_caches=2)
    generator = torch.Generator().manual_seed(7)

    sampler = _make_weighted_train_sampler(
        source_indices, normalized, num_sources=2, num_samples=8000, generator=generator
    )
    drawn = _drawn_sources(sampler, source_indices)

    frac_source_0 = sum(source == 0 for source in drawn) / len(drawn)
    assert frac_source_0 == pytest.approx(0.8, abs=0.05)


def test_samples_per_epoch_controls_sampler_length() -> None:
    source_indices = [0] * 4 + [1] * 4
    _, normalized = _normalize_cache_weights((1.0, 1.0), num_caches=2)
    generator = torch.Generator().manual_seed(0)

    sampler = _make_weighted_train_sampler(
        source_indices, normalized, num_sources=2, num_samples=5, generator=generator
    )

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 5
    assert len(list(sampler)) == 5


def test_weighted_sampler_is_deterministic_under_seed() -> None:
    source_indices = [0] * 4 + [1] * 4
    _, normalized = _normalize_cache_weights((1.0, 2.0), num_caches=2)

    first = list(
        _make_weighted_train_sampler(
            source_indices, normalized, num_sources=2, num_samples=20, generator=torch.Generator().manual_seed(99)
        )
    )
    second = list(
        _make_weighted_train_sampler(
            source_indices, normalized, num_sources=2, num_samples=20, generator=torch.Generator().manual_seed(99)
        )
    )
    assert first == second


# --- end-to-end metadata ----------------------------------------------------------


def test_default_run_records_concat_shuffle_sampling(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    _write_cache(cache_dir, num_rows=6, seed=11)

    metrics = run_train_eval(_train_args(cache_path=str(cache_dir), output_dir=str(output_dir), seed=11))

    sampling = metrics["train_sampling"]
    assert sampling["mode"] == "concat_shuffle"
    assert sampling["cache_weights"] == []
    assert sampling["cache_weights_normalized"] == []
    assert sampling["samples_per_epoch"] is None
    assert sampling["cache_weight_seed"] is None

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["train_sampling"]["mode"] == "concat_shuffle"
    # Backward-compatible train_caches shape: no weight keys when weighting is off.
    assert checkpoint["train_caches"] == [{"path": str(cache_dir), "num_samples": 6}]
    assert list(checkpoint["args"]["cache_weights"]) == []
    assert checkpoint["args"]["samples_per_epoch"] is None
    assert checkpoint["args"]["cache_weight_seed"] is None


def test_weighted_run_records_metadata_and_per_cache_weights(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    _write_cache(primary_dir, num_rows=6, seed=21)
    _write_cache(extra_dir, num_rows=2, seed=22)
    _write_cache(eval_dir, num_rows=2, seed=23)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            eval_cache_path=str(eval_dir),
            output_dir=str(output_dir),
            cache_weights=(1.0, 3.0),
            samples_per_epoch=5,
            epochs=1,
            seed=21,
        )
    )

    sampling = metrics["train_sampling"]
    assert sampling["mode"] == "weighted"
    assert sampling["cache_weights"] == pytest.approx([1.0, 3.0])
    assert sampling["cache_weights_normalized"] == pytest.approx([0.25, 0.75])
    assert sampling["samples_per_epoch"] == 5
    assert sampling["cache_weight_seed"] == 21 + 3  # default seed offset

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["train_caches"] == [
        {"path": str(primary_dir), "num_samples": 6, "weight": 1.0, "weight_normalized": 0.25},
        {"path": str(extra_dir), "num_samples": 2, "weight": 3.0, "weight_normalized": 0.75},
    ]
    assert checkpoint["train_sampling"]["cache_weight_seed"] == 24
    assert list(checkpoint["args"]["cache_weights"]) == [1.0, 3.0]


def test_weighted_run_uses_explicit_cache_weight_seed(tmp_path) -> None:
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    _write_cache(primary_dir, num_rows=4, seed=31)
    _write_cache(extra_dir, num_rows=4, seed=32)
    _write_cache(eval_dir, num_rows=2, seed=33)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            eval_cache_path=str(eval_dir),
            output_dir=str(output_dir),
            cache_weights=(1.0, 1.0),
            cache_weight_seed=4242,
            seed=31,
        )
    )

    assert metrics["train_sampling"]["cache_weight_seed"] == 4242


def test_weighted_run_normalization_uses_unweighted_union(tmp_path) -> None:
    # Even with a skewed sampler, action-normalization stats must come from the
    # unweighted train union: mean = (6*100 + 2*108)/8 = 102 per dim.
    primary_dir = tmp_path / "primary"
    extra_dir = tmp_path / "extra"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    _write_cache(primary_dir, num_rows=6, seed=41, action_value=100.0)
    _write_cache(extra_dir, num_rows=2, seed=42, action_value=108.0)
    _write_cache(eval_dir, num_rows=2, seed=43, action_value=50.0)

    metrics = run_train_eval(
        _train_args(
            cache_path=str(primary_dir),
            extra_cache_path=(str(extra_dir),),
            eval_cache_path=str(eval_dir),
            output_dir=str(output_dir),
            batch_size=4,
            normalize_actions=True,
            cache_weights=(1.0, 9.0),
            seed=41,
        )
    )

    assert metrics["train_sampling"]["mode"] == "weighted"
    assert metrics["action_normalization_mean"] == pytest.approx([102.0, 102.0, 102.0])
