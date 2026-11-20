"""Deterministic complete source passes without a 140M-element permutation array."""
import json
import math
import os
from pathlib import Path
import numpy as np


def permute_indices(indices, size, seed):
    """Cycle-walk a bijection on a power-of-two domain onto [0, size).

    Odd multiplication and xor-shifts are invertible modulo 2**bits. This is
    an inexpensive keyed shuffle, not a cryptographic or uniform permutation.
    """
    if size == 1:
        return np.zeros_like(indices, dtype=np.int64)
    bits = (int(size) - 1).bit_length()
    if bits > 32:
        raise ValueError('Permutation supports at most 2**32 records per source')
    mask = np.uint64((1 << bits) - 1)
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 1 << bits, size=6, dtype=np.uint64)
    x = np.asarray(indices, dtype=np.uint64).copy()
    active = np.arange(len(x))
    while len(active):
        v = x[active]
        for j in range(3):
            v ^= v >> np.uint64(max(1, bits // 2))
            v = (v * (keys[j * 2] | np.uint64(1)) + keys[j * 2 + 1]) & mask
        x[active] = v
        active = active[v >= size]
    return x.astype(np.int64)


def epoch_plan(sampler):
    if getattr(sampler, 'full_plan', None) is not None:
        return sampler.full_plan
    leaves = sampler.leaves
    weights = {d.name: d.sample_probability for d, _ in leaves}
    control = os.environ.get('POLAR2_SAMPLING_CONTROL')
    if control:
        weights.update(json.loads(Path(control).read_text())['weights'])
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Invalid full-epoch sampling weights')
    native = [(d, off) for d, off in leaves if d.sample_probability > 0 and weights[d.name] > 0]
    base_size = sum(len(d) for d, _ in native)
    ratios = [weights[d.name] / (len(d) / base_size) for d, _ in native]
    scale = min(ratios)
    if scale <= 0:
        raise ValueError('A full epoch cannot disable a base training source')
    plan = []
    for d, offset in leaves:
        target = int(math.ceil(base_size * weights[d.name] / scale - 1e-6))
        if d.sample_probability > 0 and weights[d.name] > 0:
            target = max(len(d), target)
        if target and len(d) < sampler.world_size:
            raise ValueError('Training source must contain at least world_size records')
        batch = d.batch_size * sampler.world_size
        if batch < 2 * sampler.world_size:
            raise ValueError('Full epochs require per-rank batch size >= 2')
        steps = math.ceil(target / batch)
        plan.append(dict(name=d.name, size=len(d), offset=offset,
                         target=target, batch=batch, steps=steps))
    sampler.full_plan = plan
    return plan


def full_epoch_batches(sampler):
    plan = epoch_plan(sampler)
    order = np.repeat(np.arange(len(plan)), [p['steps'] for p in plan])
    np.random.default_rng(sampler.seed + sampler.epoch).shuffle(order)
    # A migrated partial epoch retains each source's consumed permutation prefix.
    cursors = [p.get('start', 0) for p in plan]
    for source in order:
        p = plan[source]
        remaining = p['target'] - cursors[source]
        count = min(p['batch'], remaining)
        # Rebalance the last two batches so every training rank has a real row;
        # no row is dropped, duplicated for padding, or exceeds the batch limit.
        if 0 < remaining - count < sampler.world_size:
            count -= sampler.world_size
        start = cursors[source]
        positions = np.arange(start + sampler.rank, start + count, sampler.world_size)
        ids = np.empty(len(positions), dtype=np.int64)
        for cycle in np.unique(positions // p['size']):
            sel = positions // p['size'] == cycle
            seed = np.random.SeedSequence([sampler.seed, sampler.epoch, int(source), int(cycle)])
            ids[sel] = permute_indices(positions[sel] % p['size'], p['size'], seed)
        cursors[source] += count
        if p['name'] in os.environ.get('POLAR2_DISABLED_SOURCES', '').split(','):
            continue
        yield (ids + p['offset']).tolist()
