# Definitions used in Fourier Transforms and Series

## Fourier Transforms and Series

The fourier transform and inverse will be defined as

$$
\tilde{\phi} = \int_{\mathbb{R}^3} e^{-i\mathbf{k}\cdot \mathbf{x}} \phi(\mathbf{x})d\mathbf{x},
$$

$$
\phi = \frac{1}{(2\pi)^3}\int_{\mathbb{R}^3} e^{i\mathbf{k}\cdot \mathbf{x}} \tilde{\phi}(\mathbf{k})d\mathbf{k}
$$

For a function which is periodic on a lattice $\Lambda$ with unit cell voume $\Omega$, we instead have **fourier series**:

$$
\tilde{f}(\mathbf{k}) := \frac{(2\pi)^3}{\Omega} \int_{\Omega} e^{-i\mathbf{k}\cdot \mathbf{x}} f(\mathbf{x})d\mathbf{x}
$$

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^3}\sum_{\mathbf{k} \in \Lambda^\star} e^{i\mathbf{k}\cdot \mathbf{x}} \tilde{f}(\mathbf{k})
$$

Note the new factor of $2\pi$ in the fourier series coefficient (see also derivation below).

$\tilde{f}(\mathbf{k})$ are the fourier series coeficients and are defined only for $\mathbf{k}\in \Lambda^\star$. The above equations are the definition of the fourier series used in the code.

> ### NOTE
> $\Lambda^\star$ is the reciprocal lattice. The recirocal lattice is generated from the reciprocal cell $B$. $B$ is a $3\times3$ matrix and its columns are the reciprocal cell vectors. If the (real space) cell is $A$, with the columns being the cell vectors, then we define
> 
> $$B = 2\pi A^{-T}$$
> 
> so that the cell vectors $`\mathbf{a}_i`$ and reciprocal lattice vectors $`\mathbf{b}_i`$ satisfy $`\mathbf{a}_{i} \cdot \mathbf{b}_{j} = \delta_{ij}`$.

## Impact on Parsevals theorem

The above expression for the fourier series coefficients, involving $(2\pi)^3 / \Omega$, implies a certain form for parsevals theorem:

$$
\int_\Omega f^\star(\mathbf{r}) g(\mathbf{r}) d \mathbf{r} = \frac{\Omega}{(2\pi)^6} \sum_{\mathbf{k} \in \Lambda^\star} \tilde{f}^\star(\mathbf{k}) \tilde{g}(\mathbf{k})
$$

## Lattice of Functions

Most of the time we will have a charge density which is defined by:

$$
\rho(\mathbf{x}) = \sum_{\mathbf{u}\in \Lambda} \sum_{i=1}^{N_\text{atoms}} q_i g(\mathbf{x}-\mathbf{x}_i-\mathbf{u})
$$

Where the sum over atoms does not include periodic copies of atoms. The sum of $\mathbf{u}$ over the lattice acounts for all the infinite replicas. $g$ is some basis function defined over $\mathbb{R}^3$ and $q_i$ is a coefficient like a partial charge. We need the fourier series of $\rho$ in terms of the atom positions $\mathbf{x}_i$, the charges $q_i$ and the function $g$.

The useful result is that:

$$
\tilde{\rho}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega} \tilde{g}(\mathbf{k}) \sum_{i=1}^{N_\text{atoms}} q_i e^{-i\mathbf{k}\cdot \mathbf{x}_i}, \quad \mathbf{k}\in\Lambda^\star.
$$

Equivalently, for a single lattice sum

$$
f(\mathbf{x}) = \sum_{\mathbf{u}\in\Lambda} g(\mathbf{x}-\mathbf{u}) \quad \Rightarrow \quad \tilde{f}(\mathbf{k}) = \tilde{g}(\mathbf{k}) \frac{(2\pi)^3}{\Omega}, \quad \mathbf{k}\in\Lambda^\star.
$$

## Handling Real Valued Functions

For a real function $f$ we have

$$
\tilde{f}(\mathbf{k}) = \tilde{f}(-\mathbf{k})^\star.
$$

In code we keep only the half-space $\mathbf{k}_x > 0$ (and optionally the $\mathbf{k}=\mathbf{0}$ term). The reconstruction becomes

$$
f(\mathbf{x}) = \frac{2}{(2\pi)^3}\sum_{\mathbf{k}_x > 0}\left[\Re\{\tilde{f}(\mathbf{k})\} \cos(\mathbf{k}\cdot \mathbf{x}) - \Im\{\tilde{f}(\mathbf{k})\} \sin(\mathbf{k}\cdot \mathbf{x})\right] + \frac{1}{(2\pi)^3}\tilde{f}(\mathbf{0}).
$$

For brevity we will omit writing the $\mathbf{k}=\mathbf{0}$ term since it will be zero in most cases.


## Relation to Unnormalized FFT

Assume a real-space grid with $N = N_x N_y N_z$ points,

$$
\mathbf{x}_{\mathbf{n}} = A \frac{\mathbf{n}}{\mathbf{N}}, \quad \mathbf{n}\in\{0,\ldots,N_x-1\}\times\{0,\ldots,N_y-1\}\times\{0,\ldots,N_z-1\},
$$

and reciprocal grid

$$
\mathbf{k}_{\mathbf{m}} = B \mathbf{m}, \quad \mathbf{m}\in\{0,\ldots,N_x-1\}\times\{0,\ldots,N_y-1\}\times\{0,\ldots,N_z-1\}.
$$

Then

$$
e^{-i\mathbf{k}_{\mathbf{m}}\cdot \mathbf{x}_{\mathbf{n}}} = e^{-2\pi i\, \mathbf{m}\cdot \mathbf{n}/\mathbf{N}},
$$

so the unnormalized forward FFT output

$$
F_{\mathbf{m}} = \sum_{\mathbf{n}} f(\mathbf{x}_{\mathbf{n}}) e^{-2\pi i\, \mathbf{m}\cdot \mathbf{n}/\mathbf{N}}
$$

relates to the Fourier-series coefficients by the Riemann-sum approximation

$$
\tilde{f}(\mathbf{k}_{\mathbf{m}}) \approx \frac{(2\pi)^3}{\Omega}\left(\frac{\Omega}{N}\right) F_{\mathbf{m}} = \frac{(2\pi)^3}{N} F_{\mathbf{m}}.
$$

This matches the default unnormalized forward FFT convention in NumPy/PyTorch (inverse has the $1/N$ factor).

## Derivations

### Convention for fourier series

Let $`\mathrm{III}_\Lambda(\mathbf{x}) = \sum_{\mathbf{a}\in\Lambda} \delta(\mathbf{x}-\mathbf{a})`$ be the Dirac comb on the lattice. A periodic function can be written as a convolution of $`\mathrm{III}_\Lambda`$ with the function defined only in the unit cell ($`f_\Omega`$).

$$
f(\mathbf{x}) = f_\Omega(\mathbf{x}) \star \mathrm{III}_\Lambda(\mathbf{x}), \quad f_\Omega(\mathbf{x}) = f(\mathbf{x})\ \text{for}\ \mathbf{x}\in\Omega.
$$

Taking Fourier transforms gives

$$
\tilde{f}(\mathbf{k}) = \tilde{f}_\Omega(\mathbf{k})\,\widetilde{\mathrm{III}}_\Lambda(\mathbf{k}).
$$

The comb transforms to a reciprocal-lattice comb

$$
\widetilde{\mathrm{III}}_\Lambda(\mathbf{k}) = \frac{(2\pi)^3}{\Omega} \sum_{\mathbf{p}\in\Lambda^\star} \delta(\mathbf{k}-\mathbf{p}),
$$

and the unit-cell transform is

$$
\tilde{f}_\Omega(\mathbf{k}) = \int_{\Omega} e^{-i\mathbf{k}\cdot \mathbf{x}} f(\mathbf{x})\,d\mathbf{x}.
$$

Combining these gives the fourier *transform* of $f$, which is zero for all $\mathbf{k}$ except where $\mathbf{k} \in \Lambda^\star$:

$$
\tilde{f}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega} \sum_{\mathbf{p}\in\Lambda^\star} \delta(\mathbf{k}-\mathbf{p})
\int_{\Omega} e^{-i\mathbf{k}\cdot \mathbf{x}} f(\mathbf{x})\,d\mathbf{x},
$$

The natrual choice for the fourier series coefficeint is then

$$
\tilde{f}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega}
\int_{\Omega} e^{-i\mathbf{k}\cdot \mathbf{x}} f(\mathbf{x})\,d\mathbf{x}
$$

Subsituting into the fourier transform (the expression with $`\delta(\mathbf{k}-\mathbf{p})`$) into the inverse FT at the top of this page then gives the corresponding recosntruction formula in terms of the fourier seres coefficieints.

### Fourier series of Lattice of functions

For a lattice of translated copies,

$$
f(\mathbf{x}) = \sum_{\mathbf{u}\in\Lambda} g(\mathbf{x}-\mathbf{u}),
$$

we can use the same maths, but instead of using $f_\Omega$, the restriction of $f$ to $\Omega$, just use $g$ in its place. The result is that:

$$
\tilde{f}(\mathbf{k}) = \tilde{g}(\mathbf{k}) \frac{(2\pi)^3}{\Omega}, \quad \mathbf{k}\in\Lambda^\star.
$$

in which $\tilde g$ is the normal fourier transform of $g$ over $\mathbb{R}^3$ as defined above.
