import pytest
import torch

from kaby.data.timeseries import IrregularSineWaveDataset
from kaby.models.ode_rnn import GRUTimeSeriesBaseline, ODERNN


def _build_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    dataset = IrregularSineWaveDataset(
        num_samples=batch_size,
        n_context_points=6,
        n_future_points=4,
        min_observed_context_points=2,
        observation_prob=0.6,
        noise_std=0.01,
        seed=123,
    )
    samples = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0].keys()
    }


def test_irregular_sine_dataset_masks_are_consistent() -> None:
    dataset = IrregularSineWaveDataset(
        num_samples=4,
        n_context_points=8,
        n_future_points=5,
        min_observed_context_points=2,
        observation_prob=0.6,
        seed=7,
    )
    sample = dataset[0]

    assert sample["context_mask"].shape == (8, 1)
    assert sample["interp_mask"].shape == (8, 1)
    assert sample["future_mask"].shape == (5, 1)

    assert sample["context_mask"].sum().item() >= 2
    assert sample["interp_mask"].sum().item() >= 1

    assert torch.all(sample["context_times"][1:] > sample["context_times"][:-1])
    assert torch.all(sample["full_times"][1:] > sample["full_times"][:-1])
    assert torch.allclose(sample["full_times"][:8], sample["context_times"])

    assert sample["future_mask"].all()
    assert sample["observed_context"].shape == (8, 1)
    assert sample["context_values"].shape == (8, 1)
    assert sample["ground_truth"].shape == (13, 1)


def test_ode_rnn_shape_consistency() -> None:
    batch = _build_batch()
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.shape == batch["ground_truth"].shape


def test_gru_baseline_shape_consistency() -> None:
    batch = _build_batch()
    model = GRUTimeSeriesBaseline(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.shape == batch["ground_truth"].shape


def test_ode_rnn_nfe_tracking() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    assert model.get_nfe() == 0

    _ = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert model.get_nfe() > 0


def test_ode_rnn_gradient_flow() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )
    loss = predictions.sum()
    loss.backward()

    has_gradients = any(parameter.grad is not None for parameter in model.parameters())
    assert has_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_ode_rnn_device_agnostic() -> None:
    device = torch.device("cuda:0")
    batch = {
        key: value.to(device)
        for key, value in _build_batch(batch_size=2).items()
    }
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    ).to(device)

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.device == device