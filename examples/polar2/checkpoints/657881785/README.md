# MACE-POLAR-2 checkpoint: 657,881,785 configurations

This is the latest rolling checkpoint from the OMol/OPoly E/F training run,
saved on 2026-11-14. It is a research snapshot in the personal fork
`SunZichen-2004/mace`, based on `ACEsuit/mace`, and is not an official upstream
MACE release. This directory records the matching training source and artifact
hashes. Model/optimizer tensors are GitHub Release assets, not Git blobs.

## Download

```bash
gh release download mace-polar-2-657881785 --repo SunZichen-2004/mace \
  --pattern 'polar2_657881785_*' --dir models
(cd models && sha256sum -c ../examples/polar2/checkpoints/657881785/SHA256SUMS)
```

The [Release](https://github.com/SunZichen-2004/mace/releases/tag/mace-polar-2-657881785)
contains:

- `polar2_657881785_training.pt`: the original rolling resume, including training
  weights, optimizer/EMA state and rank-local sampling/RNG progress. Its custom
  format is `mace-walltime-resume-v1`, not an inference model.
- `polar2_657881785_schedulefree_eval.model`: serialized model for inference;
  schedule-free `optimizer.eval()` was applied exactly once during export.
  It does not use EMA weights. Do not apply the optimizer transformation again.

`manifest.json` records file sizes, SHA-256 digests, architecture and source hashes.
The raw checkpoint and inference export intentionally have different weights.

## Inference

Install this PR's code and the bundled polar dependency (the original environment
used Python 3.12, PyTorch 2.11.0+cu130, e3nn 0.4.4, graph-longrange 0.4.4).
The run also used a modified `graph_electrostatics` checkout. Its matching
source, tests and MIT license are bundled in `external/graph_electrostatics`.
The published graph-longrange 0.4.4 wheel alone is insufficient (it lacks
`RealSpaceAnalyticalElectrostaticFeatures`). From the repository root, install:

```bash
python -m pip install -e './external/graph_electrostatics'
python -m pip install -e '.[schedulefree,cueq]' scipy
```

The export contains cuEquivariance modules from the NVIDIA B200 training run;
install compatible `cuequivariance`, `cuequivariance-torch` and CUDA ops for GPU
inference. The example uses the existing converted model, without converting it
again. A stock upstream installation alone does not contain the custom classes.

```python
from ase import Atoms
from mace.calculators import MACECalculator

atoms = Atoms('H2O', positions=[[0, 0, 0], [0.9572, 0, 0],
                              [-0.2399872, 0.927297, 0]])
atoms.info.update(charge=0, spin=1)  # spin is multiplicity: 2S+1
atoms.calc = MACECalculator(
    model_paths='models/polar2_657881785_schedulefree_eval.model',
    model_type='PolarMACE', head='omol', device='cuda',
    default_dtype='float64', enable_cueq=False,
)
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

Although the serialized model has both `omol` and `bms25` heads, this run trained
OMol and OPoly E/F only; do not treat the BMS25 head as a validated ML/MM model.

## Training provenance and resumption

`training_config.original.yaml` preserves the exact original configuration,
including machine-local data/model/output paths. Those paths are provenance,
not bundled datasets. `train_with_walltime.py` is the original atomic resume and
walltime wrapper. It restores optimizer, EMA, RNG and per-rank batch progress;
`mace_run_train --restart_latest` alone is not a replacement for this custom
resume format. Reproducing/resuming training additionally requires the original
foundation model, dataset descriptors/data and compatible allocation settings.
Configure the local paths before using the wrapper. The checkpoint was saved
with world size 8; changing world size requires an explicit sampling-state
migration. No training job is launched by this PR.

## Validation scope and known issue

This exact 657,881,785 checkpoint is newer than the latest recorded validation:

- Full validation last completed at epoch 5, 623,119,033 configurations:
  OMol E/F MAE 1.73 meV/atom / 11.90 meV/Angstrom; OPoly 0.63 / 8.33.
- Latest fixed 4,096 OMol-valid evaluation was at 655,000,249 configurations:
  schedule-free E/F MAE 1.5898189034 / 11.6584937867;
  EMA 1.6490760959 / 11.6427997845, in the same units.

These are results for those checkpoints, not measurements of this release.

The source intentionally preserves the historical validation behavior for
reproducibility: `Polar2Metrics` computes force component MSE for validation
loss, whereas `POLAR2Loss` trains with mean per-atom force-error vector L2 norm.
The recorded best epoch 4 is selected by the former criterion. Epoch 5 OMol
reported force RMSE 881.01 meV/Angstrom despite force MAE 11.90; per-structure
re-evaluation is needed to establish the cause. This snapshot does not claim
to fix that discrepancy or resolve the high-RMSE anomaly.

## Verification for this snapshot

- 18 focused MACE tests passed (epoch sampling, gradient handling and reporting).
- 13 focused bundled-dependency tests passed (dtype, analytical selection,
  gradients, energy/feature consistency and graph isolation).
- The training checkpoint and inference export both load on CPU with the
  bundled source; state-dict keys match and all 16,648,867 model parameters
  are finite. No GPU inference or full-dataset re-evaluation was run for this PR.
- Both release artifacts match the recorded SHA-256 hashes.
