"""Muon for MACE radial MLP matrices, with AdamW for all other parameters."""

import logging

import torch


class MuonAdamW(torch.optim.Optimizer):
    """Expose native Muon and AdamW as one checkpointable optimizer.

    MACE stores many equivariant weights as packed vectors or tensors. Only
    the ordinary matrices in ``interactions.*.conv_tp_weights`` use Muon;
    embeddings, product bases, readouts and polar modules retain AdamW.
    Both optimizers share this object's groups and state, so the existing
    schedulers, gradient clipping and checkpoint code need no special cases.
    """

    def __init__(
        self,
        param_groups,
        lr,
        muon_lr=None,
        momentum=0.95,
        ns_steps=5,
        adjust_lr_fn="original",
        betas=(0.9, 0.999),
        amsgrad=True,
    ):
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise ImportError(
                "--optimizer=muon requires PyTorch >= 2.9 (torch.optim.Muon)"
            )
        if lr < 0 or (muon_lr is not None and muon_lr < 0):
            raise ValueError("Muon and AdamW learning rates must be non-negative")
        if ns_steps < 1 or ns_steps >= 100:
            raise ValueError("--muon_ns_steps must be between 1 and 99")

        groups = []
        for original in param_groups:
            params = list(original["params"])
            names = original.get("param_names", [])
            if len(names) != len(params):
                raise ValueError(
                    "Muon requires parameter names from get_params_options"
                )
            is_matrix = [
                name.startswith("interactions.")
                and ".conv_tp_weights." in name
                and name.endswith(".weight")
                and param.ndim == 2
                for name, param in zip(names, params)
            ]
            for use_muon in (True, False):
                selected = [
                    i for i, eligible in enumerate(is_matrix) if eligible == use_muon
                ]
                if not selected:
                    continue
                group = dict(original)
                group["params"] = [params[i] for i in selected]
                group["param_names"] = [names[i] for i in selected]
                group["use_muon"] = use_muon
                group["name"] = original.get("name", "group") + (
                    "_muon" if use_muon else "_adamw"
                )
                if use_muon and muon_lr is not None:
                    # Keep --lr_params_factors, including groups frozen with lr=0.
                    group["lr"] = muon_lr * original.get("lr", lr) / lr if lr else 0.0
                groups.append(group)

        if not any(group["use_muon"] for group in groups):
            raise ValueError(
                "Muon found no 2D interactions.*.conv_tp_weights.*.weight parameters"
            )
        super().__init__(groups, defaults={})
        self._optimizers = []
        for use_muon in (True, False):
            selected = [
                group for group in self.param_groups if group["use_muon"] == use_muon
            ]
            if not selected:
                continue
            if use_muon:
                optimizer = muon_cls(
                    selected,
                    lr=lr if muon_lr is None else muon_lr,
                    momentum=momentum,
                    ns_steps=ns_steps,
                    adjust_lr_fn=adjust_lr_fn,
                )
            else:
                optimizer = torch.optim.AdamW(
                    selected, lr=lr, betas=betas, amsgrad=amsgrad
                )
            optimizer.state = self.state
            self._optimizers.append((use_muon, optimizer))
        logging.info(
            "Muon: %s radial MLP matrices; AdamW: %s other parameter tensors",
            sum(len(g["params"]) for g in self.param_groups if g["use_muon"]),
            sum(len(g["params"]) for g in self.param_groups if not g["use_muon"]),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for _, optimizer in self._optimizers:
            optimizer.step()
        return loss

    def load_state_dict(self, state_dict):
        if [g.get("use_muon") for g in state_dict["param_groups"]] != [
            g["use_muon"] for g in self.param_groups
        ]:
            raise ValueError(
                "Checkpoint Muon/AdamW parameter groups do not match this model"
            )
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict replaces the groups and state dictionaries.
        for use_muon, optimizer in self._optimizers:
            optimizer.param_groups = [
                group for group in self.param_groups if group["use_muon"] == use_muon
            ]
            optimizer.state = self.state
