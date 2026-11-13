"""Optional source-sampling control from fixed validation monitors (rank zero only)."""
import json
import logging
import os
from pathlib import Path


def update_sampling(metrics, epoch, source):
    path = os.environ.get('POLAR2_SAMPLING_CONTROL')
    if not path:
        return
    path = Path(path)
    state = json.loads(path.read_text())
    if source.endswith('/spice_alpha'):
        name = 'alpha'
        keys = ['mae_alpha_field_A3', 'rmse_alpha_field_A3',
                'mae_alpha_multipole_A3', 'rmse_alpha_multipole_A3']
    elif source.endswith('/omol4m'):
        name = 'spin_dipole'
        keys = ['mae_spin_dipole', 'rmse_spin_dipole']
    else:
        return
    values = {k: float(metrics[k]) for k in keys}
    monitor = state.setdefault('monitors', {}).setdefault(name, {})
    if 'baseline' not in monitor:
        monitor.update(baseline=values, streak=0, last_epoch=None)
    elif epoch is not None and monitor.get('last_epoch') != epoch:
        ratios = [values[k] / max(monitor['baseline'][k], 1e-12) for k in keys]
        # Both MAE and RMSE must degrade; for alpha, either physical branch qualifies.
        bad = any(ratios[j] > 1.2 and ratios[j+1] > 1.2 for j in range(0, len(keys), 2))
        monitor['streak'] = monitor.get('streak', 0) + 1 if bad else 0
        if monitor['streak'] >= 3:
            weights = state['weights']
            target = 'spice_alpha' if name == 'alpha' else 'omol4m'
            total = sum(weights.values())
            old = weights.get(target, 0.) / total
            new = min(.10 if name == 'alpha' else .20, max(old * 2, .03 if name == 'spin_dipole' else 0))
            if new > old:
                # Fix the requested source fraction; preserve the ratios of other sources.
                other = sum(v for k, v in weights.items() if k != target)
                state['weights'] = {k: (new if k == target else v / other * (1-new)) for k, v in weights.items()}
                event = dict(epoch=epoch, monitor=name, old_fraction=old, new_fraction=new, metrics=values)
                state.setdefault('events', []).append(event)
                logging.info('POLAR-2 sampling adjustment: %s', event)
            monitor['streak'] = 0
        monitor['last_epoch'] = epoch
    monitor['latest'] = values
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)
