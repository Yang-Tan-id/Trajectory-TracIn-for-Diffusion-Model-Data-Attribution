"""X3 diffusion task and checkpoint adapter for simple-influence SOURCE."""

import sys

import torch
from torch.utils.data import Dataset

import x3pixel_DM_training as base
from source_das_config import *


if not (SOURCE_DAS_SIMPLE_INFLUENCE_ROOT / "src" / "source.py").is_file():
    raise FileNotFoundError(
        "simple-influence was not found at "
        f"{SOURCE_DAS_SIMPLE_INFLUENCE_ROOT}; set SIMPLE_INFLUENCE_ROOT"
    )
sys.path.insert(0, str(SOURCE_DAS_SIMPLE_INFLUENCE_ROOT))

from src.abstract_task import AbstractTask  # noqa: E402
from src.source import SourceComputer  # noqa: E402


class TimestampAlignedTrainDataset(Dataset):
    def __init__(self, dataset, family, noises):
        self.dataset = dataset
        self.family = family
        self.noises = noises

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x0, condition = self.dataset[index]
        if self.family == "unprompted":
            condition = torch.zeros_like(condition)
        return {
            "x0": x0,
            "condition": condition,
            "noises": self.noises[index],
        }


class TrajectoryOutputComponentDataset(Dataset):
    def __init__(self, query_records, trajectories, conditions, snapshot_index):
        self.query_records = query_records
        self.trajectories = trajectories
        self.conditions = conditions
        self.snapshot_index = int(snapshot_index)
        self.output_dim = SOURCE_DAS_OUTPUT_DIM

    def __len__(self):
        return len(self.query_records) * self.output_dim

    def __getitem__(self, index):
        query_index = int(index) // self.output_dim
        component = int(index) % self.output_dim
        xt = torch.from_numpy(
            self.trajectories[query_index][self.snapshot_index, 0]
        ).to(torch.float32)
        return {
            "xt": xt,
            "condition": self.conditions[query_index].squeeze(0).cpu(),
            "output_index": torch.tensor(component, dtype=torch.long),
        }


class X3TimestampSourceTask(AbstractTask):
    def __init__(self, timestamp, schedule, device):
        super().__init__(device=device)
        self.timestamp = int(timestamp)
        self.schedule = schedule

    def _call_model(self, model, parameters, x, t, condition):
        if parameters is None:
            return model(x, t, condition)
        params, buffers = parameters
        return torch.func.functional_call(model, (params, buffers), (x, t, condition))

    def get_train_loss(
        self,
        model,
        batch,
        parameter_and_buffer_dicts=None,
        sample=False,
        reduction="sum",
    ):
        if sample:
            raise ValueError("SOURCE-DAS uses timestamp-aligned empirical Fisher")
        x0 = batch["x0"].to(self.device)
        condition = batch["condition"].to(self.device)
        noises = batch["noises"].to(self.device)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)
            condition = condition.unsqueeze(0)
            noises = noises.unsqueeze(0)
        batch_size, mc = noises.shape[:2]
        x_mc = x0[:, None].expand(batch_size, mc, *x0.shape[1:]).reshape(
            batch_size * mc, *x0.shape[1:]
        )
        condition_mc = condition[:, None].expand(
            batch_size, mc, condition.shape[-1]
        ).reshape(batch_size * mc, condition.shape[-1])
        noise_flat = noises.reshape(batch_size * mc, *noises.shape[2:])
        timesteps = torch.full(
            (batch_size * mc,), self.timestamp, device=self.device, dtype=torch.long
        )
        xt = base.q_sample(x_mc, timesteps, noise_flat, self.schedule)
        prediction = self._call_model(
            model, parameter_and_buffer_dicts, xt, timesteps, condition_mc
        )
        per_example = (
            (prediction - noise_flat).pow(2).reshape(batch_size, mc, -1)
            .mean(dim=2).mean(dim=1)
        )
        if reduction == "none":
            return per_example
        if reduction == "mean":
            return per_example.mean()
        if reduction == "sum":
            return per_example.sum()
        raise ValueError(f"unsupported reduction={reduction!r}")

    def get_measurement(
        self,
        model,
        batch,
        parameter_and_buffer_dicts=None,
        sample=False,
        reduction="sum",
    ):
        del sample
        xt = batch["xt"].to(self.device)
        condition = batch["condition"].to(self.device)
        output_index = batch["output_index"].to(self.device)
        if xt.ndim == 3:
            xt = xt.unsqueeze(0)
            condition = condition.unsqueeze(0)
            output_index = output_index.unsqueeze(0)
        timesteps = torch.full(
            (xt.shape[0],), self.timestamp, device=self.device, dtype=torch.long
        )
        prediction = self._call_model(
            model, parameter_and_buffer_dicts, xt, timesteps, condition
        ).reshape(xt.shape[0], -1)
        selected = prediction.gather(1, output_index.reshape(-1, 1)).squeeze(1)
        if reduction == "none":
            return selected
        if reduction == "mean":
            return selected.mean()
        if reduction == "sum":
            return selected.sum()
        raise ValueError(f"unsupported reduction={reduction!r}")

    def get_batch_size(self, batch):
        if "x0" in batch:
            return int(batch["x0"].shape[0])
        return int(batch["xt"].shape[0])

    def influence_modules(self):
        return list(SOURCE_DAS_INFLUENCE_MODULES)

    def representation_module(self):
        return "out_conv"


class X3SourceComputer(SourceComputer):
    """SOURCE computer that reads the X3 checkpoint payload format."""

    def _load_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model = self.model.to(self.task.device)
        self.model.eval()
