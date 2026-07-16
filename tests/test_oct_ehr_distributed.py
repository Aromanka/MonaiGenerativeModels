from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from oct_ehr_ldm.config import ProjectConfig
from oct_ehr_ldm.cli import _base_parser
from oct_ehr_ldm.data import DistributedEvalSampler, create_loader
from oct_ehr_ldm.runtime import (
    DistributedContext,
    all_reduce_sum,
    distributed_session,
    gather_rng_states,
    wrap_ddp,
)


class TestDistributedDataLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = TensorDataset(torch.arange(12))
        self.config = ProjectConfig(
            {
                "project_root": ".",
                "data": {"batch_size": 2, "num_workers": 0},
                "training": {"seed": 17},
            },
            Path("test-config.json"),
        )

    def test_eval_sampler_shards_without_duplicates(self) -> None:
        shards = [set(DistributedEvalSampler(self.dataset, rank, 3)) for rank in range(3)]
        self.assertEqual(set.union(*shards), set(range(len(self.dataset))))
        self.assertFalse(shards[0] & shards[1])
        self.assertFalse(shards[0] & shards[2])
        self.assertFalse(shards[1] & shards[2])

    def test_torchrun_local_rank_argument_is_accepted_before_or_after_command(self) -> None:
        before = _base_parser().parse_args(["--local-rank", "2", "train-autoencoder"])
        after = _base_parser().parse_args(["train-autoencoder", "--local_rank", "3"])
        self.assertEqual(before.local_rank, 2)
        self.assertEqual(after.local_rank, 3)

    @unittest.skipIf(os.name == "nt", "This Windows PyTorch build cannot create a Gloo device")
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_process_ddp_collectives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            rendezvous = (Path(temp) / "ddp-store").resolve().as_uri()
            mp.spawn(_spawn_ddp_worker, args=(2, rendezvous), nprocs=2, join=True)

    def test_training_loader_uses_rank_specific_distributed_sampler(self) -> None:
        shards: list[set[int]] = []
        for rank in range(3):
            context = DistributedContext(torch.device("cpu"), rank=rank, local_rank=rank, world_size=3)
            loader = create_loader(self.dataset, self.config, training=True, epoch=4, distributed=context)
            self.assertIsInstance(loader.sampler, DistributedSampler)
            shards.append(set(loader.sampler))
        self.assertEqual(set.union(*shards), set(range(len(self.dataset))))
        self.assertFalse(shards[0] & shards[1])
        self.assertFalse(shards[0] & shards[2])
        self.assertFalse(shards[1] & shards[2])


def _assert_ddp_operations(context: DistributedContext) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(context.rank + 1)
    wrapped = wrap_ddp(model, context)
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))

    value = torch.tensor([[float(context.rank + 1)]])
    wrapped(value).sum().backward()
    expected_gradient = torch.full_like(model.weight.grad, (context.world_size + 1) / 2)
    torch.testing.assert_close(model.weight.grad, expected_gradient)

    reduced = torch.tensor(float(context.rank + 1))
    all_reduce_sum(reduced, context)
    expected_sum = context.world_size * (context.world_size + 1) / 2
    torch.testing.assert_close(reduced, torch.tensor(expected_sum))
    assert len(gather_rng_states(context)) == context.world_size


def _spawn_ddp_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        context = DistributedContext(torch.device("cpu"), rank=rank, local_rank=rank, world_size=world_size)
        _assert_ddp_operations(context)
    finally:
        dist.destroy_process_group()


def _run_torchrun_smoke_test() -> None:
    with distributed_session("cpu") as context:
        _assert_ddp_operations(context)


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_torchrun_smoke_test()
    else:
        unittest.main()
