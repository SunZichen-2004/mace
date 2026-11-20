"""Constant-memory, exact distributed sums for complete POLAR-2 validation."""
from collections import defaultdict
import torch

SPECS = [
    ('energy', 'energy', 'e', True),
    ('forces', 'forces', 'f', False),
    ('MM_forces', 'MM_forces', 'MM_forces', False),
    ('dipole', 'dipole', 'mu', False),
    ('dipole', 'total_dipole_from_electric_field', 'mu_field', False),
    ('quadrupole', 'quadrupole', 'quadrupole', False),
    ('spin_dipole', 'spin_dipole', 'spin_dipole', False),
    ('spin_quadrupole', 'spin_quadrupole', 'spin_quadrupole', False),
    ('polarizability', 'polar2_alpha_field', 'alpha_field_A3', False),
    ('polarizability', 'polar2_alpha_multipole', 'alpha_multipole_A3', False),
]


class Polar2Metrics:
    def __init__(self, loss_fn, device):
        self.loss_fn = loss_fn
        # count, abs, square, normalized abs, normalized square, weighted loss
        self.sums = torch.zeros((len(SPECS), 6), dtype=torch.float64, device=device)
        self.num_data = torch.zeros((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, batch, output):
        self.num_data += batch.num_graphs
        n = (batch.ptr[1:] - batch.ptr[:-1]).double()
        for row, (name, branch, _, absolute) in enumerate(SPECS):
            # The source can carry labels for tasks disabled in this E/F run.
            # Match POLAR2Loss.weighted_components: absent zero-weight branches
            # neither require predictions nor contribute validation metrics.
            if float(getattr(self.loss_fn, name + '_weight')) == 0.0:
                continue
            target = getattr(batch, name)
            if name in ('quadrupole', 'spin_quadrupole'):
                target = target.reshape(-1, 3, 3)
            mask = getattr(batch, name + '_weight').double()
            if name in ('forces', 'MM_forces'):
                ids = batch.batch if name == 'forces' else batch.MM_batch
                mask = mask.reshape(-1)[ids] * batch.weight[ids]
            else:
                mask = mask * batch.weight.reshape((-1,) + (1,) * (mask.ndim - 1))
            while mask.ndim < target.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(target)
            active = mask != 0
            if not active.any():
                continue
            if output.get(branch) is None:
                raise ValueError(f'Missing active validation prediction {branch}')
            pred = output[branch].double()
            if name == 'polarizability':
                pred = pred * 14.3996454784255
            delta = pred - target.double()
            normalized = delta if name in ('forces', 'MM_forces') else delta / n.reshape((-1,) + (1,) * (delta.ndim - 1))
            d, nd = delta[active], normalized[active]
            values = torch.stack((active.sum(), d.abs().sum(), d.square().sum(),
                                  nd.abs().sum(), nd.square().sum(),
                                  (mask[active] * (nd.abs() if absolute else nd.square())).sum()))
            self.sums[row] += values

    def compute(self):
        values = self.sums.clone()
        count = self.num_data.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
            torch.distributed.all_reduce(count)
        aux = defaultdict(lambda: None)
        parts = {name: 0.0 for name, *_ in SPECS}
        for row, (name, _, metric, _) in enumerate(SPECS):
            n, a, s, na, ns, weighted = values[row].tolist()
            if n == 0:
                continue
            aux['mae_' + metric] = a / n
            aux['rmse_' + metric] = (s / n) ** .5
            if name in ('energy', 'dipole'):
                aux['mae_' + metric + '_per_atom'] = na / n
                aux['rmse_' + metric + '_per_atom'] = (ns / n) ** .5
            branches = 2 if name in ('dipole', 'polarizability') else 1
            parts[name] += float(getattr(self.loss_fn, name + '_weight')) * weighted / n / branches
        aux.update({'loss_' + k: v for k, v in parts.items()})
        aux['loss'] = sum(parts.values())
        aux['num_configs'] = int(count.item())
        return aux['loss'], aux
