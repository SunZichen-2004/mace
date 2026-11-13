# MACE-POLAR-2 and the BMS25 ML/MM checkpoint

This branch contains PolarMACE response models, external point-charge ML/MM
embedding, charge/spin field options, and the accompanying training and tests.
See `mace/modules/field_block_equations.md` for the response equations.

The separate model repository is
[SunZichen-2004/MACE-POLAR-2-ML-MM-model](https://github.com/SunZichen-2004/MACE-POLAR-2-ML-MM-model).
Its `step-14500` release preserves the exact checkpoint used for the nine-protein,
185-conformation miniprotein comparison. It provides the inference model, full
training-state snapshot, runtime sources, SHA256 checksums and benchmark values.

That checkpoint is the **five-width Pairwise variant**: the MACE-POLAR-1-M
geometry backbone with two `PairwiseGatedMultipoleUpdate` steps and the original
`OneBodyMLPFieldReadout`. Receive widths are `[0.5, 0.8, 1.2, 1.5, 3.0]` Å,
density width is 1.5 Å, Fukui regularisation is 0.1, and the field is recomputed
after the final polarization update. It uses analytical real-space electrostatics
and the `alpha_beta` field representation. The optional `charge_spin`/screened
spin-field mode in this branch was **not enabled** in this checkpoint.

All parameters were fine-tuned on BMS25; atomic energy offsets were fitted on
2,000 parent-training conformations. The model started from pretrained shared
weights and newly initialized Pairwise response blocks, not a previously trained
5,000-frame model. The miniprotein category was held out from training and
validation. The published comparison uses Cartesian-component gradient MAE and
all unordered within-protein conformation pairs for relative energy. Exact
reference pairing with the AMP-BMS25 paper has not been established.

Use the release's archived runtime for exact historical reproduction; the branch
also contains development features that postdate that training run. The archived
inference model already has schedule-free evaluation weights. Do not apply a
second optimizer-to-evaluation conversion to it.
