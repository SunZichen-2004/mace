import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset
from mace.data.polar2_dataset import Polar2Dataset, Polar2BatchSampler
from mace.data.polar2_epoch import permute_indices, epoch_plan
from mace.tools.polar2_metrics import Polar2Metrics
from mace.modules.polar2_loss import POLAR2Loss
from types import SimpleNamespace


def source(name, n, batch, probability):
    d = Polar2Dataset.__new__(Polar2Dataset)
    d.name, d.ends, d.limit = name, np.array([n]), None
    d.batch_size, d.sample_probability = batch, probability
    return d


@pytest.mark.parametrize('size', [1, 2, 3, 17, 32, 1031, 10003])
def test_bijective_shuffle(size):
    ids = permute_indices(np.arange(size), size, 33)
    assert np.array_equal(np.sort(ids), np.arange(size))
    assert np.array_equal(ids[::7], permute_indices(np.arange(size)[::7], size, 33))


def test_full_epoch_covers_each_record_once_with_disjoint_rank_tails(monkeypatch):
    monkeypatch.delenv('POLAR2_SAMPLING_CONTROL', raising=False)
    data = ConcatDataset([source('a', 1031, 8, 1031/1398), source('b', 367, 4, 367/1398)])
    batches = []
    for rank in range(16):
        sampler = Polar2BatchSampler(data, -1, rank, 16, 91)
        sampler.set_epoch(2)
        rows = list(sampler)
        assert len(rows) == len(sampler)
        assert all(1 <= len(x) <= (8 if x[0] < 1031 else 4) for x in rows)
        batches.append(rows)
    ids = [i for rank in batches for row in rank for i in row]
    assert sorted(ids) == list(range(len(data)))
    for step in zip(*batches):
        ids = [i for rank in step for i in rank]
        assert len(ids) == len(set(ids))
        assert len(set(i < 1031 for i in ids)) == 1
    sampler.set_epoch(3)
    assert list(sampler) != batches[15]


def test_validation_has_no_padding_no_missing_tail_or_collective_requirement():
    data = source('a', 137, 3, 1)
    ids = []
    for rank in range(16):
        sampler = Polar2BatchSampler(data, rank=rank, world_size=16, validation=True)
        rows = list(sampler)
        assert len(rows) == len(sampler)
        ids += [i for row in rows for i in row]
    assert sorted(ids) == list(range(137))


def test_adaptive_oversampling_preserves_full_base_pass(tmp_path, monkeypatch):
    import json
    control = tmp_path/'control.json'
    control.write_text(json.dumps({'weights': {'a': .5, 'b': .4, 'spin': .1}}))
    monkeypatch.setenv('POLAR2_SAMPLING_CONTROL', str(control))
    data = ConcatDataset([source('a', 800, 8, .8), source('b', 200, 4, .2), source('spin', 100, 4, 0)])
    sampler = Polar2BatchSampler(data, -1, world_size=2)
    assert [p['target'] for p in epoch_plan(sampler)] == [800, 640, 160]
    all_ids=[]
    for rank in range(2):
        sampler.rank=rank
        all_ids.extend(i for row in sampler for i in row)
    assert set(range(1000)).issubset(all_ids)
    assert len(all_ids)==1600


def batch_for_metrics(n):
    b=SimpleNamespace(ptr=torch.arange(n+1)*2,batch=torch.arange(n).repeat_interleave(2),
        MM_batch=torch.empty(0,dtype=torch.long),weight=torch.ones(n),num_graphs=n)
    for name, shape in dict(energy=(n,),forces=(2*n,3),MM_forces=(0,3),dipole=(n,3),
            quadrupole=(n,3,3),spin_dipole=(n,3),spin_quadrupole=(n,3,3),polarizability=(n,3,3)).items():
        setattr(b,name,torch.zeros(shape));setattr(b,name+'_weight',torch.zeros(n))
    b.energy_weight[:]=1;b.forces_weight[:]=1
    return b


def test_streaming_exact_loss_and_metrics_are_batch_partition_independent():
    loss=POLAR2Loss()
    def accumulate(sizes):
        m=Polar2Metrics(loss,'cpu');start=0
        for n in sizes:
            b=batch_for_metrics(n)
            e=torch.arange(start+1,start+n+1,dtype=torch.float32)
            f=e.repeat_interleave(2)[:,None].expand(-1,3)
            m.update(b,{'energy':e,'forces':f});start+=n
        return m.compute()
    a=accumulate([7]);b=accumulate([3,3,1])
    assert dict(a[1])==dict(b[1])
    assert a[1]['num_configs']==7
    assert a[1]['mae_e_per_atom']==2
    assert a[0]==pytest.approx(10*2+10*20)
