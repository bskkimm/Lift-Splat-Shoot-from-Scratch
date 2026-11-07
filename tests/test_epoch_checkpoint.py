import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


def trained_components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.zero_grad()
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_epoch_checkpoint_restores_optimizer_scheduler_and_random_streams(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(14)
    model, optimizer, scheduler = trained_components()
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    config = {"split": "train", "batch_size": 2, "seed": 13}
    checkpoint = tmp_path / "epoch.pt"

    save_epoch_checkpoint(checkpoint, model, optimizer, epoch=1, config=config,
                          loader_generator=generator, scheduler=scheduler)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=generator)
    with torch.no_grad():
        model.weight.add_(5)
    optimizer.param_groups[0]["lr"] = 0.5
    scheduler.step()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    generator.manual_seed(99)

    epoch = load_epoch_checkpoint(checkpoint, model, optimizer, config=config,
                                  loader_generator=generator, scheduler=scheduler)

    assert epoch == 1
    for key, value in expected_model.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key in expected_optimizer["state"]:
        for name, value in expected_optimizer["state"][key].items():
            torch.testing.assert_close(actual_optimizer["state"][key][name], value)
    assert scheduler.state_dict() == expected_scheduler
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    torch.testing.assert_close(torch.rand(3, generator=generator), expected_loader)


def test_checkpoint_rejects_config_mismatch_before_mutating_model(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    model, optimizer, scheduler = trained_components()
    generator = torch.Generator().manual_seed(1)
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, model, optimizer, 1, {"split": "train"}, generator, scheduler)
    with torch.no_grad():
        model.weight.add_(4)
    before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="config"):
        load_epoch_checkpoint(checkpoint, model, optimizer, {"split": "val"}, generator, scheduler)
    torch.testing.assert_close(model.weight, before)


def test_checkpoint_requires_matching_scheduler_presence(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    model, optimizer, scheduler = trained_components()
    generator = torch.Generator().manual_seed(1)
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, model, optimizer, 1, {}, generator, scheduler)
    with pytest.raises(ValueError, match="scheduler"):
        load_epoch_checkpoint(checkpoint, model, optimizer, {}, generator)


def test_checkpoint_rejects_incompatible_optimizer_before_mutation(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    model, optimizer, _ = trained_components()
    generator = torch.Generator().manual_seed(1)
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, model, optimizer, 1, {}, generator)
    with torch.no_grad():
        model.weight.add_(4)
    before = model.weight.detach().clone()
    incompatible = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="optimizer"):
        load_epoch_checkpoint(checkpoint, model, incompatible, {}, generator)
    torch.testing.assert_close(model.weight, before)


def test_checkpoint_rejects_model_shape_mismatch_before_any_weight_copy(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    source = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    source_optimizer = torch.optim.Adam(source.parameters())
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, source, source_optimizer, 1, {}, torch.Generator())
    target = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    target_optimizer = torch.optim.Adam(target.parameters())
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(ValueError, match="model.*shape"):
        load_epoch_checkpoint(checkpoint, target, target_optimizer, {}, torch.Generator())
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


def test_checkpoint_rejects_optimizer_group_mismatch_before_weight_copy(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, model, source_optimizer, 1, {}, torch.Generator())
    with torch.no_grad():
        model.weight.add_(4)
    before = model.weight.detach().clone()
    target_optimizer = torch.optim.Adam([{"params": [model.weight]}, {"params": [model.bias]}])

    with pytest.raises(ValueError, match="optimizer.*groups"):
        load_epoch_checkpoint(checkpoint, model, target_optimizer, {}, torch.Generator())
    torch.testing.assert_close(model.weight, before)


class ModelWithVoxelPool(torch.nn.Module):
    def __init__(self, x_bounds):
        super().__init__()
        from lss.models.voxel_pool import VoxelPool

        self.linear = torch.nn.Linear(2, 1)
        self.pool = VoxelPool(x_bounds, (0, 1, 1), (0, 1, 1))


def test_failed_extra_state_restore_rolls_back_model_weights(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    source = ModelWithVoxelPool((0, 1, 0.5))
    source_optimizer = torch.optim.Adam(source.parameters())
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, source, source_optimizer, 1, {}, torch.Generator())
    target = ModelWithVoxelPool((1, 2, 0.5))
    target_optimizer = torch.optim.Adam(target.parameters())
    before = target.linear.weight.detach().clone()

    with pytest.raises((ValueError, RuntimeError), match="bounds"):
        load_epoch_checkpoint(checkpoint, target, target_optimizer, {}, torch.Generator())
    torch.testing.assert_close(target.linear.weight, before, rtol=0, atol=0)


def test_late_rng_restore_failure_rolls_back_all_previous_state(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    model, optimizer, scheduler = trained_components()
    generator = torch.Generator().manual_seed(1)
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, model, optimizer, 1, {}, generator, scheduler)
    state = torch.load(checkpoint, weights_only=True)
    state["torch_rng"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(state, checkpoint)
    with torch.no_grad():
        model.weight.add_(4)
    optimizer.param_groups[0]["lr"] = 0.25
    scheduler.last_epoch = 19
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    generator.manual_seed(94)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    before_loader = generator.get_state().clone()

    with pytest.raises(RuntimeError):
        load_epoch_checkpoint(checkpoint, model, optimizer, {}, generator, scheduler)

    for key, value in before_model.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert optimizer.state_dict()["param_groups"] == before_optimizer["param_groups"]
    for key in before_optimizer["state"]:
        for name, value in before_optimizer["state"][key].items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)
    assert scheduler.state_dict() == before_scheduler
    assert random.getstate() == before_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == before_numpy[0] and actual_numpy[2:] == before_numpy[2:]
    np.testing.assert_array_equal(actual_numpy[1], before_numpy[1])
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), before_loader, rtol=0, atol=0)


def test_failed_checkpoint_write_preserves_existing_file(tmp_path, monkeypatch):
    from lss.training_checkpoint import save_epoch_checkpoint

    model, optimizer, scheduler = trained_components()
    checkpoint = tmp_path / "epoch.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        save_epoch_checkpoint(checkpoint, model, optimizer, 1, {}, torch.Generator(), scheduler)
    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["epoch.pt"]


class RandomEpochDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        noise = random.random() + float(np.random.random()) + torch.rand(()).item()
        return torch.tensor([float(index), noise]), torch.tensor([float(index % 2)])


def make_epoch_run():
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    generator = torch.Generator().manual_seed(34)
    loader = DataLoader(RandomEpochDataset(), batch_size=2, shuffle=True,
                        generator=generator, num_workers=0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler, generator, loader


def run_epoch(model, optimizer, scheduler, loader):
    model.train()
    for inputs, targets in loader:
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    scheduler.step()


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint

    config = {"split": "synthetic", "batch_size": 2, "workers": 0, "shuffle": True}
    model, optimizer, scheduler, generator, loader = make_epoch_run()
    run_epoch(model, optimizer, scheduler, loader)
    run_epoch(model, optimizer, scheduler, loader)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    interrupted, interrupted_optimizer, interrupted_scheduler, interrupted_generator, interrupted_loader = make_epoch_run()
    run_epoch(interrupted, interrupted_optimizer, interrupted_scheduler, interrupted_loader)
    checkpoint = tmp_path / "epoch.pt"
    save_epoch_checkpoint(checkpoint, interrupted, interrupted_optimizer, 1, config,
                          interrupted_generator, interrupted_scheduler)

    resumed, resumed_optimizer, resumed_scheduler, resumed_generator, resumed_loader = make_epoch_run()
    assert load_epoch_checkpoint(checkpoint, resumed, resumed_optimizer, config,
                                 resumed_generator, resumed_scheduler) == 1
    run_epoch(resumed, resumed_optimizer, resumed_scheduler, resumed_loader)

    for key, expected in expected_model.items():
        torch.testing.assert_close(resumed.state_dict()[key], expected, rtol=0, atol=0)
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key in expected_optimizer["state"]:
        for name, expected in expected_optimizer["state"][key].items():
            torch.testing.assert_close(actual_optimizer["state"][key][name], expected, rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == expected_scheduler


def test_checkpoint_round_trip_loads_assembled_lss_model(tmp_path):
    from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
    from tests.test_lss_segmentation import make_model, synthetic_inputs

    torch.manual_seed(123)
    model = make_model().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(5)
    images, intrinsics, extrinsics = synthetic_inputs()
    with torch.no_grad():
        expected = model(images, intrinsics, extrinsics)
    checkpoint = tmp_path / "lss.pt"
    save_epoch_checkpoint(checkpoint, model, optimizer, 1, {"recipe": "synthetic"}, generator)

    restored = make_model().eval()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    restored_generator = torch.Generator()
    assert load_epoch_checkpoint(checkpoint, restored, restored_optimizer,
                                 {"recipe": "synthetic"}, restored_generator) == 1
    with torch.no_grad():
        actual = restored(images, intrinsics, extrinsics)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
