# GTO basis: normalization and Fourier transform

We use a Gaussian type orbital (GTO) basis of the form

$$
\phi_{nlm}(\mathbf{r}) = C_{l,\sigma}\,e^{-\frac{r^2}{2\sigma^2}} r^{l} Y_{lm}(\hat{\mathbf{r}}),
$$

where $Y_{lm}$ are real spherical harmonics. The code follows the e3nn convention
(`normalize=True`, `normalization="integral"`), so

$$
\int Y_{lm}(\hat{\mathbf{r}})\,Y_{LM}(\hat{\mathbf{r}})\,d\hat{\mathbf{r}}
= \delta_{lL}\delta_{mM},\quad Y_{00}=1/\sqrt{4\pi}.
$$

## Multipole normalization constant

Define the regular solid harmonics

$$
R_{lm}(\mathbf{r}) = \sqrt{\frac{4\pi}{2l+1}}\, r^{l} Y_{lm}(\hat{\mathbf{r}}),
$$

and the multipole moments

$$
Q_{lm} = \int \rho(\mathbf{r}) R_{lm}(\mathbf{r})\,d\mathbf{r}.
$$

The only nonzero moment of $\phi_{nlm}$ is the matching $(l,m)$ component:

$$
\bar{Q}_{lm} = \int \phi_{nlm}(\mathbf{r}) R_{lm}(\mathbf{r})\,d\mathbf{r}.
$$

Using

$$
\int_0^\infty e^{-\frac{r^2}{2\sigma^2}} r^{2l+2}\,dr
= 2^{\frac{2l+1}{2}} \Gamma\!\left(\frac{2l+3}{2}\right)\sigma^{2l+3},
$$

we obtain

$$
\bar{Q}_{lm}
= C_{l,\sigma}\,\sqrt{\frac{4\pi}{2l+1}}
\,2^{\frac{2l+1}{2}}
\Gamma\!\left(\frac{2l+3}{2}\right)\sigma^{2l+3}.
$$

If we want unit multipole moment for each basis function, we choose

$$
C_{l,\sigma} =
\left[
\sqrt{\frac{4\pi}{2l+1}}
\,2^{\frac{2l+1}{2}}
\Gamma\!\left(\frac{2l+3}{2}\right)\sigma^{2l+3}
\right]^{-1}.
$$

This is the normalization used when the code selects `"multipoles"`.

## Fourier transform of a GTO

With the Fourier transform convention

$$
\tilde{\phi}(\mathbf{k}) = \int_{\mathbb{R}^3} e^{-i\mathbf{k}\cdot\mathbf{r}}
\phi(\mathbf{r})\,d\mathbf{r},
$$

the GTO transform is

$$
\tilde{\phi}_{nlm}(\mathbf{k}) =
4\pi\,C_{l,\sigma}\,(-i)^l\,Y_{lm}(\hat{\mathbf{k}})
\int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}}\,dr.
$$

Define the radial factor

$$
f_{l,\sigma}(k)
= 4\pi \int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}}\,dr,
$$

so

$$
\tilde{\phi}_{nlm}(\mathbf{k})
= C_{l,\sigma} (-i)^l Y_{lm}(\hat{\mathbf{k}})\,f_{l,\sigma}(k).
$$

The closed form is

$$
f_{l,\sigma}(k)
= 4\pi \sqrt{\frac{\pi}{2}}\,k^l \sigma^{3+2l}\,
{}_1F_1\!\left(\frac{3}{2}+l,\frac{3}{2}+l,-\frac{(k\sigma)^2}{2}\right).
$$

For even $l$, $\tilde{\phi}$ is purely real; for odd $l$, it is purely imaginary, with
the phase factor set by $(-i)^l$.
