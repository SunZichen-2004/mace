"""Exact masked logical-batch loss with bounded activation memory."""
import os, math
from contextlib import nullcontext
import torch
from mace.data.polar2_dataset import collate_graphs
from .polar2_gradients import backward_scale, clip_scaled_gradients


def accumulated_backward(model, loss_fn, batch, optimizer, output_args, max_grad_norm, captured):
    size = int(os.environ['POLAR2_MICROBATCH_SIZE'])
    if getattr(batch, 'source_name', None) and batch.source_name[0] == 'opoly26':
        size = min(size, int(os.environ.get('POLAR2_MICROBATCH_SIZE_OPOLY26', '32')))
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    chunks = torch.tensor(math.ceil(batch.num_graphs / size), device=batch.energy.device)
    if distributed:
        torch.distributed.all_reduce(chunks, op=torch.distributed.ReduceOp.MAX)
    chunks = int(chunks)
    if batch.num_graphs < chunks:
        raise ValueError('Cannot form nonempty microbatches on every rank')
    groups = torch.tensor_split(torch.arange(batch.num_graphs), chunks)
    from copy import copy
    from mace.tools.torch_geometric.data import Data
    unpack = copy(batch)
    unpack.__data_class__ = Data  # AtomicData requires constructor arguments.
    graphs = unpack.to_data_list()
    counts = loss_fn.component_counts(batch)
    optimizer.zero_grad(set_to_none=True)
    accumulated = {}; outputs = {}; common_scale = 1.0
    keep = {'energy','forces','MM_forces','dipole','total_dipole_from_electric_field',
            'quadrupole','spin_dipole','spin_quadrupole','polar2_alpha_field','polar2_alpha_multipole'}
    for index, group in enumerate(groups):
        micro = collate_graphs([graphs[i] for i in group.tolist()])
        kwargs = dict(training=True, compute_force=output_args['forces'],
                      compute_virials=output_args['virials'], compute_stress=output_args['stress'])
        kwargs = loss_fn.forward_kwargs(micro, kwargs)
        context = model.no_sync() if hasattr(model, 'no_sync') and index < chunks - 1 else nullcontext()
        with context:
            output = model(micro.to_dict(), **kwargs)
            parts = loss_fn.weighted_components(micro, output, global_counts=counts)
            loss = sum(parts.values())
            scale = min(common_scale, backward_scale(loss))
            if scale < common_scale:
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale / common_scale)
            common_scale = scale
            (loss * common_scale).backward()
        for key, value in parts.items():
            accumulated[key] = accumulated.get(key, 0) + value.detach()
        for key in keep:
            if output.get(key) is not None:
                outputs.setdefault(key, []).append(output[key].detach())
        del output, parts, loss, micro
    captured['parts'] = accumulated
    captured['output'] = {k: torch.cat(v, dim=0) for k, v in outputs.items()}
    captured['microbatches'] = chunks
    captured['microbatch_size'] = size
    captured['backward_scale'] = common_scale
    captured['gradient_norm_before_clip'] = clip_scaled_gradients(model.parameters(), common_scale, max_grad_norm)
    return sum(accumulated.values())
