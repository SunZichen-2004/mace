"""Run MACE with atomic epoch resumes and coordinated Slurm walltime shutdown.

The rolling resume is separate from MACE's best-validation checkpoints. Saves
happen every configured number of global configurations, at epoch boundaries,
and on an early walltime stop. Pending validation is preserved on resume.
"""
import functools
import importlib
import logging
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch


class WalltimeStop(Exception):
    pass


class RemainingBatches:
    def __init__(self, sampler, skip):
        self.sampler, self.skip = sampler, skip

    def __iter__(self):
        for index, batch in enumerate(self.sampler):
            if index >= self.skip:
                yield batch

    def __len__(self):
        return max(0, len(self.sampler) - self.skip)


def atomic_save(payload, path):
    """The old file remains valid until a fully flushed replacement is ready."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_resume(payload, path):
    path = Path(path)
    if path.exists():
        previous = path.with_suffix('.previous.pt')
        temporary_link = path.with_suffix('.previous.tmp')
        temporary_link.unlink(missing_ok=True)
        os.link(path, temporary_link)
        os.replace(temporary_link, previous)
    atomic_save(payload, path)


def load_resume(path, device):
    path = Path(path)
    for candidate in (path, path.with_suffix('.previous.pt')):
        if not candidate.exists():
            continue
        try:
            value = torch.load(candidate, map_location=device, weights_only=False)
            if value.get('format') != 'mace-walltime-resume-v1':
                raise ValueError(f'Invalid resume format: {candidate}')
            return value
        except Exception:
            logging.exception('Cannot load %s; trying previous rolling resume', candidate)
    if path.exists() or path.with_suffix('.previous.pt').exists():
        raise RuntimeError('Rolling resumes exist but neither is loadable; refusing a silent fresh start')
    return None


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def reset_ema_from_model(ema, model):
    from torch_ema import ExponentialMovingAverage
    fresh = ExponentialMovingAverage(model.parameters(), decay=ema.decay, use_num_updates=True)
    ema.load_state_dict(fresh.state_dict())


def install_guard():
    if os.environ.get('NODE_DATA_DIR'):
        from node_cache import install_verified_cache
        install_verified_cache()
    import mace.tools as tools
    from mace.tools.checkpoint import CheckpointIO
    train_module = importlib.import_module('mace.tools.train')
    if os.environ.get('MACE_STAGE_TIMING_DIR'):
        from stage_timing import install_reader_timing, install_stage_timing
        install_reader_timing()
        install_stage_timing(train_module)
    original_train = tools.train
    original_epoch = train_module.train_one_epoch
    original_step = train_module.take_step

    # Protect ordinary best-validation checkpoints too. Delete the old file only
    # after the new checkpoint has been flushed and committed successfully.
    def atomic_best_save(self, checkpoint, epochs, keep_last=False):
        path = os.path.join(self.directory, self._get_checkpoint_filename(epochs, self.swa_start))
        atomic_save(checkpoint, path)
        if not self.keep and self.old_path and not keep_last and self.old_path != path:
            Path(self.old_path).unlink(missing_ok=True)
        self.old_path = path
    CheckpointIO.save = atomic_best_save

    @functools.wraps(original_train)
    def guarded_train(**kwargs):
        kwargs['valid_loaders'] = {k: v for k, v in kwargs['valid_loaders'].items() if k in ('omol/omol140m', 'omol/opoly26')}
        train_leaves = kwargs['train_sampler'].leaves
        if any(leaf.split != 'train' for leaf, _ in train_leaves):
            raise RuntimeError('Validation or test split appeared in the training sampler')
        from mace.data.polar2_dataset import dataset_leaves
        for loader in kwargs['valid_loaders'].values():
            if any(leaf.split != 'valid' for leaf, _ in dataset_leaves(loader.dataset)):
                raise RuntimeError('Non-validation split appeared in a validation loader')
        model, optimizer = kwargs['model'], kwargs['optimizer']
        scheduler, ema = kwargs['lr_scheduler'], kwargs['ema']
        rank, device = kwargs['rank'], kwargs['device']
        distributed = kwargs['distributed']
        handler = kwargs['checkpoint_handler']
        path = Path(handler.io.directory) / f'{handler.io.tag}_resume.pt'
        deadline = float(os.environ.get('MACE_TRAIN_STOP_AT', 'inf'))
        stopped = [False]
        current_epoch = [kwargs['start_epoch']]
        completed_batches = [0]
        seen_configs = [0]
        last_checkpoint_configs = [0]
        validation_pending = [False]
        last_lr_control_id = [0]
        active_sampler = [kwargs['train_sampler']]
        interval = int(os.environ.get('POLAR2_CHECKPOINT_CONFIGS', '1000000'))
        resume = load_resume(path, device)
        if resume is not None:
            model.load_state_dict(resume['model'], strict=True)
            if resume.get('reset_optimizer_once', False):
                assert not optimizer.state, 'Expected freshly initialized optimizer'
                logging.info('PROTOCOL_SWITCH: loaded current model weights; optimizer and scheduler freshly initialized; warmup restarts; EMA retained')
            else:
                optimizer.load_state_dict(resume['optimizer'])
                scheduler.load_state_dict(resume['lr_scheduler'])
            logging.info('Optimizer startup: state_entries=%s; groups=%s', len(optimizer.state),
                         [{k: group.get(k) for k in ('lr', 'k', 'warmup_steps', 'train_mode')} for group in optimizer.param_groups])
            if ema is not None and resume.get('reset_ema_once', False):
                reset_ema_from_model(ema, model)
                verification = dict(num_updates=ema.num_updates, decay=ema.decay,
                    shadow_equals_current_parameters=all(torch.equal(a, b.detach()) for a, b in zip(ema.shadow_params, model.parameters())),
                    optimizer_steps=[g.get('k') for g in optimizer.param_groups],
                    seen_configs=resume.get('seen_configs'), rank=rank)
                assert verification['num_updates'] == 0 and verification['shadow_equals_current_parameters']
                if rank == 0:
                    (Path(__file__).parent / 'ema_reset_verified.json').write_text(json.dumps(verification, indent=2))
                logging.info('EMA_RESET_VERIFIED: %s', verification)
            elif ema is not None and resume['ema'] is not None:
                ema.load_state_dict(resume['ema'])
                logging.info('EMA_RESTORED: num_updates=%s, decay=%s', ema.num_updates, ema.decay)
            kwargs['start_epoch'] = resume['next_epoch']
            current_epoch[0] = resume['next_epoch']
            if resume.get('seen_configs', 0) > 0:
                os.environ['MACE_SKIP_INITIAL_EVAL'] = '1'
            seen_configs[0] = resume.get('seen_configs', 0)
            last_checkpoint_configs[0] = resume.get('last_checkpoint_configs', seen_configs[0])
            validation_pending[0] = resume.get('validation_pending', False)
            last_lr_control_id[0] = resume.get('last_lr_control_id', 0)
            if resume['world_size'] != (torch.distributed.get_world_size() if distributed else 1):
                raise RuntimeError('Exact partial-epoch resume requires the same number of ranks')
            logging.info('Rolling resume: next_epoch=%s, batches_completed=%s, completed_epoch=%s',
                         resume['next_epoch'], resume['batches_completed'], resume['epoch_complete'])
        model.compute_dipole_from_electric_field = False
        model.compute_spin_multipoles = False
        model.compute_polarizability = False
        assert tuple(model.dipole_cap) == (2.0, 4.0), getattr(model, 'dipole_cap', None)
        logging.info('Verified resumed dipole_cap=%s; disabled training sources=%s', model.dipole_cap, os.environ.get('POLAR2_DISABLED_SOURCES'))
        logging.info('Walltime guard: stop_at_unix=%s; rolling checkpoint=%s; save each 1000000 global configurations and on epoch/stop boundaries', deadline, path)

        def mark_stop(signum, frame):
            stopped[0] = True
        old_handlers = {s: signal.signal(s, mark_stop) for s in (signal.SIGTERM, signal.SIGUSR1)}

        memory_last = [0.0]
        def memory_probe(phase):
            if rank % 8 or time.time() - memory_last[0] < 10:
                return
            memory_last[0] = time.time()
            cg = Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_'+os.environ.get('POLAR2_ALLOCATION','837370'))
            try:
                stat = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
                limit = int((cg/'memory.max').read_text())
                # File-backed pages are reclaimable; tmpfs/shmem is not.
                used = sum(int(stat.get(k, 0)) for k in ('anon','shmem','kernel'))
                record = dict(time=time.time(), rank=rank, phase=phase, node=os.uname().nodename,
                              working_bytes=used, limit_bytes=limit,
                              current_bytes=int((cg/'memory.current').read_text()),
                              seen_configs=seen_configs[0], batches_completed=completed_batches[0])
                target=Path(__file__).parent / ('memory_'+os.uname().nodename+'.json')
                temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record));temp.replace(target)
                if used > .85 * limit:
                    logging.error('HOST_MEMORY_PRESSURE: working=%s limit=%s; requesting coordinated checkpoint',used,limit)
                    stopped[0] = True
            except (OSError, ValueError) as exc:
                logging.warning('Host memory probe failed: %s',exc)

        def should_stop():
            memory_probe('training')
            # Every rank participates at the same optimizer-step boundary.
            flag = torch.tensor([int(stopped[0] or time.time() >= deadline)], device=device)
            if distributed:
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())

        def snapshot(epoch_complete, archive=False):
            rng = capture_rng()
            world_size = torch.distributed.get_world_size() if distributed else 1
            all_rng = [None] * world_size if rank == 0 else None
            if distributed:
                torch.distributed.gather_object(rng, all_rng, dst=0)
            else:
                all_rng = [rng]
            ok = torch.ones(1, device=device)
            if rank == 0:
                try:
                    payload = dict(format='mace-walltime-resume-v1', model=model.state_dict(),
                                   optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                                   ema=ema.state_dict() if ema is not None else None,
                                   next_epoch=current_epoch[0] + int(epoch_complete),
                                   epoch_complete=epoch_complete,
                                   batches_completed=0 if epoch_complete else completed_batches[0],
                                   world_size=world_size, rng=all_rng,
                                   protocol='full-coverage-v1',
                                   seen_configs=seen_configs[0],
                                   last_checkpoint_configs=last_checkpoint_configs[0],
                                   validation_pending=validation_pending[0],
                                   last_lr_control_id=last_lr_control_id[0],
                                   sampler_plan=getattr(active_sampler[0], 'full_plan', None))
                    save_resume(payload, path)
                    if archive:
                        directory = Path(os.environ.get('POLAR2_INTERVAL_CHECKPOINT_DIR', path.parent / 'interval'))
                        directory.mkdir(parents=True, exist_ok=True)
                        destination = directory / f'{handler.io.tag}_configs-{seen_configs[0]:012d}.pt'
                        # May be a different filesystem from the rolling checkpoint.
                        atomic_save(payload, destination)
                        logging.info('Configuration checkpoint: seen=%s; path=%s', seen_configs[0], destination)
                    logging.info('Committed rolling resume: next_epoch=%s, batches_completed=%s, epoch_complete=%s',
                                 payload['next_epoch'], payload['batches_completed'], epoch_complete)
                except Exception:
                    logging.exception('Rolling checkpoint write failed; previous committed checkpoint retained')
                    ok.zero_()
            if distributed:
                torch.distributed.broadcast(ok, src=0)
            if not ok.item():
                raise RuntimeError('Rolling checkpoint write failed on rank 0')

        def guarded_epoch(**epoch_kwargs):
            nonlocal resume
            current_epoch[0] = epoch_kwargs['epoch']
            completed_batches[0] = 0
            skip = 0
            if resume is not None:
                if resume.get('sampler_plan') is not None and not resume['epoch_complete']:
                    sampler = epoch_kwargs['data_loader'].batch_sampler
                    saved = resume['sampler_plan']
                    if len(saved) != len(sampler.leaves):
                        raise RuntimeError('Resume source list changed')
                    for entry, (leaf, _) in zip(saved, sampler.leaves):
                        if entry['name'] != leaf.name or entry['size'] != len(leaf) or entry['batch'] != leaf.batch_size * sampler.world_size:
                            raise RuntimeError('Cannot replay partial epoch with changed data or batch sizes')
                    sampler.full_plan = saved
                skip = resume['batches_completed']
                completed_batches[0] = skip
                restore_rng(resume['rng'][rank])
                resume = None
            if should_stop():
                raise WalltimeStop()
            active_sampler[0] = epoch_kwargs['data_loader'].batch_sampler
            if validation_pending[0]:
                logging.info('Resuming pending complete validation for full epoch %s', current_epoch[0])
                return
            if skip:
                loader = epoch_kwargs['data_loader']
                options = dict(dataset=loader.dataset, batch_sampler=RemainingBatches(loader.batch_sampler, skip),
                               num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                               pin_memory=loader.pin_memory, worker_init_fn=loader.worker_init_fn,
                               generator=loader.generator, persistent_workers=loader.persistent_workers)
                if loader.num_workers:
                    options['prefetch_factor'] = loader.prefetch_factor
                epoch_kwargs['data_loader'] = torch.utils.data.DataLoader(**options)
                logging.info('Resume epoch %s: skipping %s already-updated batches per rank', current_epoch[0], skip)
            original_epoch(**epoch_kwargs)
            # This is outside the EMA evaluation context, with schedulefree in
            # training mode: raw weights agree with optimizer state.
            validation_pending[0] = True
            snapshot(epoch_complete=False)

        def after_validation(epoch):
            if 'ScheduleFree' in type(optimizer).__name__:
                optimizer.train()
            validation_pending[0] = False
            snapshot(epoch_complete=True)

        def eval_should_stop():
            memory_probe("validation")
            return stopped[0] or time.time() >= deadline

        def eval_abort():
            raise WalltimeStop()

        def guarded_step(**step_kwargs):
            # Rank 0 decides at a shared step boundary, then broadcasts the
            # same control to every optimizer replica before the update.
            if completed_batches[0] % 32 == 0:
                control = None
                if rank == 0 and os.environ.get('POLAR2_LR_CONTROL'):
                    control_path = Path(os.environ['POLAR2_LR_CONTROL'])
                    if control_path.exists():
                        control = json.loads(control_path.read_text())
                if distributed:
                    message = [control]
                    torch.distributed.broadcast_object_list(message, src=0)
                    control = message[0]
                if control is not None and int(control['id']) > last_lr_control_id[0]:
                    old = float(control['expected_lr'])
                    new = float(control['new_lr'])
                    if not (new > 0 and abs(new - old * 0.5) < 1e-12):
                        raise RuntimeError(f'Invalid LR reduction control: {control}')
                    for group in optimizer.param_groups:
                        if abs(float(group['lr']) - old) > 1e-12:
                            raise RuntimeError(f'LR control expected {old}, got {group["lr"]}')
                        group['lr'] = new
                    last_lr_control_id[0] = int(control['id'])
                    logging.info('VALIDATION_LR_REDUCTION: %s', json.dumps(control, sort_keys=True))
                    if rank == 0:
                        with (Path(__file__).parent / 'lr_applied.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(control=control, applied_seen_configs=seen_configs[0],
                                                         time=time.time())) + '\n')
            result = original_step(**step_kwargs)
            completed_batches[0] += 1
            count = torch.tensor(step_kwargs['batch'].num_graphs, dtype=torch.int64, device=device)
            if distributed:
                torch.distributed.all_reduce(count)
            seen_configs[0] += int(count.item())
            if seen_configs[0] // interval > last_checkpoint_configs[0] // interval:
                last_checkpoint_configs[0] = seen_configs[0]
                snapshot(epoch_complete=False, archive=True)
            if completed_batches[0] % 32 == 0 and should_stop():
                raise WalltimeStop()
            return result

        train_module.polar2_after_validation = after_validation
        train_module.polar2_eval_should_stop = eval_should_stop
        train_module.polar2_eval_abort = eval_abort
        train_module.train_one_epoch = guarded_epoch
        train_module.take_step = guarded_step
        try:
            original_train(**kwargs)
        except WalltimeStop:
            if 'ScheduleFree' in type(optimizer).__name__:
                optimizer.train()
            snapshot(epoch_complete=False)
            logging.info('Walltime/signal stop committed safely; exiting before final full-dataset evaluation')
            if distributed:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            raise SystemExit(0)
        finally:
            train_module.polar2_after_validation = None
            train_module.polar2_eval_should_stop = None
            train_module.polar2_eval_abort = None
            train_module.train_one_epoch = original_epoch
            train_module.take_step = original_step
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)

    tools.train = guarded_train


if __name__ == '__main__':
    torch.backends.cuda.preferred_blas_library('cublas')
    install_guard()
    from mace.cli.run_train import main
    main()
