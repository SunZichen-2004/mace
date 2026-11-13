# PairwiseGatedMultipoleUpdate: gated h, F, and TP(V,h)

This specifies the new architecture, replacing the previous six-pair gate block.
The registry name remains `PairwiseGatedMultipoleUpdate`; old checkpoints of that
block are architecturally incompatible. `AgnosticEmbeddedOneBodyVariableUpdate`
and its existing checkpoints are unchanged.

## Indices and conventions

- i: atom; u: update step (different steps have independent parameters).
- s in {up,down}: spin channel; n=1,...,N_l: field radial channel.
- l,m: spherical rank and component, m=-l,...,l.
- k=1,...,C_l: hidden channel; this implementation requires C_l=C=C_0.
- K=sum_l C_l; q_l(k)=sum_{j<l} C_j+k (one-based gate indexing).
- h means the unprojected mixed backbone features h^(T), in mul_ir layout.
- All weights below are effective weights, including fixed e3nn linear/path
  normalization factors where relevant. v is the already normalized field
  feature tensor supplied by PolarMACE; no additional field normalization occurs.
- Equations assume the natural-parity, one-block-per-rank irreps used in Polar.
  Existing sparse TP supports l x l -> 0, l x 0 -> l, and 0 x l -> l only.
  In particular, the normal 0e+1o configuration is supported; these equations do
  not claim support for a full angular TP with l=2.

## Pure field and mixed embedding

\[
F^{(u)}_{i,klm}=
\sum_{n=1}^{N_l}\sum_{s\in\{\uparrow,\downarrow\}}
W^{v,(u)}_{lkns}v^{(u),s}_{i,nlm}.
\]

\[
V^{(u)}_{i,klm}=F^{(u)}_{i,klm}
+\sum_s W^{p,(u)}_{lks}p^{(u),s}_{i,lm}
+\sum_{k'=1}^{C_l}W^{h,(u)}_{lkk'}h_{i,k'lm}
+c^{(u)}_k\delta_{l0}\delta_{m0}.
\]

F has no bias. Its projection weights are shared with the F term inside V.
The scalar embedding bias c is implemented by h_linear(biases=True); only 0e
outputs receive a bias. There is no extra h projection on the direct h path.

## Original scalar contraction and element embedding

Unlike the paper's l-resolved descriptor, the existing sparse contraction sums
all ranks into C scalar channels. Preserve that behavior:

\[
D^{(u)}_{i,k}=\sum_l\sum_{k'=1}^{C_l}
\frac{\omega^{(u)}_{lkk'}}{\sqrt{2l+1}}
\sum_{m=-l}^{l}h_{i,klm}V^{(u)}_{i,k'lm},\qquad k=1,\ldots,C.
\]

Here omega includes the sparse TP's fixed path weight.

\[
e^{(u)}_{i,q}=\sum_{Z=1}^{N_{\rm elem}}E^{(u)}_{qZ}\mathbf1[z_i=Z],
\qquad q=1,\ldots,C.
\]

\[
x^{(0)}_{i,r}=\begin{cases}
D^{(u)}_{i,r},&1\le r\le C,\\
e^{(u)}_{i,r-C},&C<r\le2C.
\end{cases}
\]

## Three gate families, with complete MLP sums

The existing RadialMLP includes LayerNorm before SiLU at each hidden layer.
Set M_0=2C and M_1=M_2=M_3=64. For t=1,2,3:

\[
y^{(t)}_{i,a}=\sum_{r=1}^{M_{t-1}}A^{(t,u)}_{ar}x^{(t-1)}_{i,r}
+\beta^{(t,u)}_a,\qquad a=1,\ldots,64,
\]
\[
\mu^{(t)}_i=\frac1{64}\sum_{a=1}^{64}y^{(t)}_{i,a},\qquad
(\sigma^{(t)}_i)^2=\frac1{64}\sum_{a=1}^{64}(y^{(t)}_{i,a}-\mu^{(t)}_i)^2,
\]
\[
x^{(t)}_{i,a}=\operatorname{SiLU}\left(
\gamma^{(t,u)}_a\frac{y^{(t)}_{i,a}-\mu^{(t)}_i}
{\sqrt{(\sigma^{(t)}_i)^2+10^{-5}}}+\zeta^{(t,u)}_a\right).
\]

The output is linear, without LayerNorm, sigmoid, or softmax:

\[
A^{\rm out}_{i,r}=\sum_{a=1}^{64}A^{(4,u)}_{ra}x^{(3)}_{i,a}
+\beta^{(4,u)}_r,\qquad r=1,\ldots,3K.
\]
\[
a^{h,(u)}_{i,lk}=A^{\rm out}_{i,q_l(k)},\quad
a^{F,(u)}_{i,lk}=A^{\rm out}_{i,K+q_l(k)},\quad
b^{(u)}_{i,lk}=A^{\rm out}_{i,2K+q_l(k)}.
\]

Each gate is shared across m; every output gate can depend on all D channels.

## Existing sparse tp_mix, in the exact input order (V,h)

Let P_L be the existing supported paths into rank L, obeying triangle/parity
selection and the sparse restrictions stated above. With real-basis normalized
CG coefficients and effective path weights theta:

\[
T^{(u)}_{i,kLM}=\sum_{(l_1,l_2)\in\mathcal P_L}
\sum_{k'=1}^{C_{l_2}}\theta^{(u)}_{l_1l_2L,kk'}
\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}
C^{LM}_{l_1m_1,l_2m_2}
V^{(u)}_{i,kl_1m_1}h_{i,k'l_2m_2}.
\]

uvu means output channel k follows the first input V, while the second input's
channel k' is summed. Fixed e3nn path normalization is included in theta.
For 0e+1o the actual sparse-kernel equations are:

\[
T_{i,k00}=\sum_{k'=1}^{C}\theta^{00}_{kk'}V_{i,k00}h_{i,k'00}
+\sum_{k'=1}^{C}\frac{\theta^{11}_{kk'}}{\sqrt3}
\sum_{m=-1}^{1}V_{i,k1m}h_{i,k'1m},
\]
\[
T_{i,k1m}=\sum_{k'=1}^{C}\frac{\theta^{10}_{kk'}}{\sqrt3}
V_{i,k1m}h_{i,k'00}
+\sum_{k'=1}^{C}\frac{\theta^{01}_{kk'}}{\sqrt3}
V_{i,k00}h_{i,k'1m}.
\]

Here each theta includes its instruction.path_weight; u was suppressed only in
these last two equations. There is no 1o x 1o -> 1o path (parity forbids it).

## New equivariant features

\[
g^{(u)}_{i,klm}=a^{h,(u)}_{i,lk}h_{i,klm}
+a^{F,(u)}_{i,lk}F^{(u)}_{i,klm}
+b^{(u)}_{i,lk}T^{(u)}_{i,klm}.
\]

There is no direct a^p P stream. Density p still affects both the gates through
V -> D and the TP term through V -> T.

## Existing gated readout, with channel sums

The unchanged GeneralNonLinearBiasReadoutBlock has two equivariant gated hidden
layers and a final equivariant linear map. Let r^(0)=g. Hidden scalar and
non-scalar multiplicities are 32 per natural-parity output rank; the projection
also supplies one scalar gate per non-scalar hidden channel. For t=1,2:

\[
S^{(t)}_{i,a}=\sum_{k=1}^{R^{(t-1)}_0}
U^{(t)}_{ak}r^{(t-1)}_{i,k00}+s^{(t)}_a,
\]
\[
B^{(t)}_{i,la}=\sum_{k=1}^{R^{(t-1)}_0}
G^{(t)}_{la k}r^{(t-1)}_{i,k00}+d^{(t)}_{la},\qquad l>0,
\]
\[
Y^{(t)}_{i,alm}=\sum_{k=1}^{R^{(t-1)}_l}
U^{(t)}_{lak}r^{(t-1)}_{i,klm},\qquad l>0,
\]
\[
r^{(t)}_{i,a00}=\phi_s(S^{(t)}_{i,a}),\qquad
r^{(t)}_{i,alm}=\phi_g(B^{(t)}_{i,la})Y^{(t)}_{i,alm}\quad(l>0).
\]

R_l^(0)=C_l; subsequent multiplicities follow the readout irreps. phi_s and
phi_g are the existing e3nn-second-moment-normalized SiLU and sigmoid. The first
linear projection has no biases (s^(1)=d^(1)=0); the second has 0e biases.
With eta=step_scale (default 0.25), the final channel sums are:

\[
\Delta p^{(u),s}_{i,lm}=\eta\left(
\sum_{a=1}^{R^{(2)}_l}O^{(u)}_{lsa}r^{(2)}_{i,alm}
+o^{(u)}_s\delta_{l0}\delta_{m0}\right),
\]
\[
f^{(u),s}_i=\sum_{a=1}^{R^{(2)}_0}J^{(u)}_{sa}r^{(2)}_{i,a00}+j^{(u)}_s.
\]

Only multipole increments are scaled by eta, not Fukui sources.

## Outer update and conservation (unchanged)

\[
\widetilde p^{(u),s}_{i,lm}=p^{(u),s}_{i,lm}+\Delta p^{(u),s}_{i,lm}.
\]

Let w_i^s be the existing normalized Fukui weights on the graph G containing i.
Without constraint groups, the monopole correction is:

\[
p^{(u+1),s}_{i,lm}=\widetilde p^{(u),s}_{i,lm}
+\delta_{l0}\delta_{m0}w^{(u),s}_i
\left(Q_G^s-\sum_{j\in G}\widetilde p^{(u),s}_{j,00}\right).
\]

Q_G^up=(Q_G+S_G)/2, Q_G^down=(Q_G-S_G)/2 in the existing density convention;
S_G is the spin imbalance, represented in input as total_spin-1.
For fukui_regularisation r>0, existing code uses (separately per graph/spin):

\[
M=|G|,\quad q=M^{-1}\sum_{j\in G}f_j^2,\quad
s=\sum_{j\in G}f_j,\quad A=\sum_{j\in G}\sqrt{f_j^2+10^{-6}q},
\]
\[
w_i=\frac{f_i s}{s^2+(rA)^2}
+\frac{(rA)^2}{M[s^2+(rA)^2]}.
\]

For all-zero sources use w_i=1/M. For r=0 the existing implementation uses
w_i=f_i/s, substituting denominator 1 when s=0. Configured dipole caps and
fragment/group constraints remain in the existing outer model, unmodified.

## MessagePassingMultipoleUpdate

An opt-in alternative to the atom-local Pairwise update. Its field projection
shares weights across the two concatenated spin potentials:

\[
F_{i,klm}=\sum_n W^v_{lkn}(v_{i,nlm}^{\uparrow}+v_{i,nlm}^{\downarrow}),\qquad
V_i=F_i+W^h h_i+W^p p_i+c\,\delta_{l0}\delta_{m0}.
\]

`num_channels` (default 128) sets the multiplicity of both `0e` and `1o` in
\(V\), independently of the pretrained backbone width. Exactly one existing
`RealAgnosticResidualInteractionBlock` aggregates neighbor features using the
radial MLP and the \(l\leq1\) spherical harmonics. The existing
`EquivariantProductBasisBlock` then forms the MACE symmetric self products
(`correlation=3` by default), including its species-conditioned skip connection.
A nonlinear equivariant readout produces the two spin multipole increments and
two Fukui logits. Only the increments are multiplied by `step_scale=0.25`.
The normal PolarMACE charge/spin constraints and Fukui regularisation follow.
There is one such MACE layer per polarization update; backbone features are
unchanged as inputs. The optional radial cutoff is forwarded to the interaction.
The cuEq converter also transfers these additional symmetric product weights;
the readout uses explicit layout conversions around its e3nn nonlinear layers.

Configuration:

```python
fixedpoint_update_config = {
    "type": "MessagePassingMultipoleUpdate",
    "num_channels": 128,
    "correlation": 3,
}
```
