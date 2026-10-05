\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!55!black,
  citecolor=blue!55!black,
  urlcolor=blue!55!black
}

\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\SO}{\mathrm{SO}}
\newcommand{\OO}{\mathrm{O}}
\newcommand{\Ncal}{\mathcal{N}}
\newcommand{\Ecal}{\mathcal{E}}
\newcommand{\Tcal}{\mathcal{T}}
\newcommand{\D}{\mathrm{D}}
\newcommand{\RePart}{\operatorname{Re}}
\newcommand{\ImPart}{\operatorname{Im}}
\newcommand{\scatter}{\operatorname{scatter\_sum}}
\newcommand{\MLP}{\operatorname{MLP}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\ind}{\mathbb{1}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\e}{\mathrm{e}}

\definecolor{codegreen}{rgb}{0.10,0.48,0.20}
\definecolor{codegray}{rgb}{0.40,0.40,0.40}
\definecolor{codeblue}{rgb}{0.10,0.25,0.62}
\lstset{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{codeblue},
  commentstyle=\color{codegreen},
  stringstyle=\color{codegray},
  columns=fullflexible,
  frame=single,
  breaklines=true,
  showstringspaces=false
}

\title{Bond-Centered SO(2) Density Expansions\\
with SO(3)-Equivariant Atomic Message Passing}
\author{Implementation specification for a MACE-compatible model}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This document specifies a bond-centered message-passing architecture for
interatomic potentials. Persistent hidden states live on atoms and transform as
irreducible representations of $\SO(3)$.  For each directed central bond, atomic
states are rotated from the global frame into the bond frame.  The environment
of the bond is expanded in a cylindrical basis, producing $\SO(2)$ Fourier
modes.  Neighbor contributions are summed before higher correlation orders are
formed, so the construction uses the atomic cluster expansion density trick and
never enumerates tuples of environment atoms.  The resulting bond-frame
messages are rotated back to the global frame and accumulated on atoms.  Every
interaction layer may contribute both an atomic energy and a reversal-symmetric
bond energy.  The complete model is $\SO(3)$ equivariant internally and produces
an invariant total energy, equivariant forces, and, when periodic cell degrees
of freedom are present, a covariant stress.
\end{abstract}

\tableofcontents

\section{Design goals}

The architecture is intended to satisfy the following requirements.
\begin{enumerate}[leftmargin=2em]
  \item Element identity enters only through the initial atomic node feature.
  There is no separate element-pair embedding in the geometric basis.
  \item Persistent atomic states are $\SO(3)$ equivariant and are stored in the
  same global frame used by MACE/e3nn.
  \item A central bond defines a local $z$ axis.  All environment products are
  performed with $\SO(2)$ algebra in that bond frame.
  \item A density projection is formed before products are taken.  Correlation
  order therefore does not require explicit enumeration of neighbor tuples.
  \item Each layer supports a node-energy readout, an edge-energy readout, or a
  hybrid of the two.
  \item Directed messages may be asymmetric, but every physical bond-energy
  contribution is explicitly symmetric under edge reversal.
\end{enumerate}

The intended high-level flow is
\begin{equation}
\boxed{
\text{global atomic irreps}
\longrightarrow
\text{bond-frame atomic irreps}
\longrightarrow
\text{bond density}
\longrightarrow
\text{$\SO(2)$ correlations}
\longrightarrow
\text{bond-frame message}
\longrightarrow
\text{global atomic update}.}
\end{equation}

\begin{figure}[ht]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  node distance=7mm and 6mm,
  box/.style={draw, rounded corners, align=center, minimum height=9mm,
              fill=blue!4, inner xsep=5pt},
  arr/.style={-{Latex[length=2mm]}, thick}
]
\node[box] (h) {$h_k^{(t,\ell m)}$\\global atomic state};
\node[box, right=of h] (rot) {$\D^\ell(F_{ij})$\\global $\to$ bond};
\node[box, right=of rot] (phi) {$\phi_{ij,k,q}^{(t)}$\\one-particle feature};
\node[box, right=of phi] (A) {$A_{ij,q}^{(t)}$\\density scatter};
\node[box, below=of A] (B) {$B_{ij,q}^{(t,\nu)}$\\density products};
\node[box, left=of B] (mloc) {$\widetilde m_{i\leftarrow j}^{(t,LM)}$\\local message};
\node[box, left=of mloc] (mglob) {$m_{i\leftarrow j}^{(t,LM)}$\\global message};
\node[box, left=of mglob] (update) {$h_i^{(t+1,LM)}$\\residual update};
\draw[arr] (h) -- (rot);
\draw[arr] (rot) -- (phi);
\draw[arr] (phi) -- (A);
\draw[arr] (A) -- (B);
\draw[arr] (B) -- (mloc);
\draw[arr] (mloc) -- (mglob);
\draw[arr] (mglob) -- (update);
\end{tikzpicture}
}
\caption{One interaction layer.  The geometry basis and bond rotations are
shared by all layers.}
\label{fig:flow}
\end{figure}

\section{Graph, indices, and conventions}

Atoms are indexed by $i,j,k$.  Their positions and atomic numbers are
$\bm R_i\in\R^3$ and $Z_i$.  A directed neighbor graph contains edges
\begin{equation}
  \Ecal=\{(i,j):0<r_{ij}<r_{\mathrm{cut}}^{\mathrm{edge}}\},
  \qquad
  \bm r_{ij}=\bm R_j-\bm R_i,
  \qquad
  r_{ij}=\lVert\bm r_{ij}\rVert.
\end{equation}
Both directions of every physical bond are stored.

For each directed central bond $e=(i,j)$, define a bond environment
\begin{equation}
  \Ncal(ij)=
  \left\{k\ne i,j:
  \lVert\bm R_k-\overline{\bm R}_{ij}\rVert
  <r_{\mathrm{cut}}^{\mathrm{env}}\right\},
  \qquad
  \overline{\bm R}_{ij}=\frac{\bm R_i+\bm R_j}{2}.
  \label{eq:bond-neighborhood}
\end{equation}
The implementation may replace the midpoint sphere by a smooth union of the
endpoint neighborhoods.  This changes only the cutoff factor below, not the
representation theory.  Excluding $i$ and $j$ keeps the environment expansion
separate from the explicit two-body path.

The triplet list is
\begin{equation}
  \Tcal=\{(ij,k):(i,j)\in\Ecal,\ k\in\Ncal(ij)\}.
\end{equation}
No pair $(k_1,k_2)$ or higher neighbor tuple is stored.

Channel indices are $c,c',\alpha,\beta$ for atomic / node width $C$, and
$a,n$ for the narrower bond-density width $C_\phi\le C$.  Global $\SO(3)$
angular momenta are $\ell,L$ with magnetic indices $m,M$.  Cylindrical
geometry modes are $\mu$, and general bond-frame $\SO(2)$ orders are $q,p$.
Correlation order is $\nu$.  The index $n$ on the cylindrical radial
embedding is an MLP output channel in the $C_\phi$ space, not a Bessel /
longitudinal basis label.

We use complex spherical and Fourier components in the derivation because the
selection rules are transparent.  A production implementation may use the real
irreps and real Wigner matrices already used by MACE/e3nn; Section
\ref{sec:real} gives the conversion.

\section{Atomic state and element initialization}

At interaction layer $t$, atom $i$ carries a direct sum of $\SO(3)$ irreps,
\begin{equation}
  h_i^{(t)}
  =\left\{h_{i,c}^{(t,\ell m)}:
  0\le \ell\le \ell_{\max},\ -\ell\le m\le\ell,
  1\le c\le C_\ell\right\}.
  \label{eq:atomic-state}
\end{equation}
Under a global rotation $Q\in\SO(3)$,
\begin{equation}
  h_{i,c}^{(t,\ell m)}
  \longmapsto
  \sum_{m'=-\ell}^{\ell}
  \D_{mm'}^\ell(Q)h_{i,c}^{(t,\ell m')}.
\end{equation}

Element information appears only in the initial scalar node embedding:
\begin{equation}
  h_{i,c}^{(0,00)}=\mathrm{Embed}_{c}(Z_i),
  \qquad
  h_{i,c}^{(0,\ell m)}=0\quad(\ell>0).
  \label{eq:element-init}
\end{equation}
The embedding table is the only module that directly reads $Z_i$.  Subsequent
chemical dependence is carried by the evolving atomic states.  In particular,
radial functions and product weights do not have separate $(Z_i,Z_j)$ lookup
tables.

An optional atomic reference energy can be implemented as a linear readout of
$h_i^{(0,00)}$.  It therefore respects the same rule and does not introduce an
additional element input.

\section{Bond frames and cylindrical coordinates}

For directed bond $(i,j)$ define
\begin{equation}
  \widehat{\bm z}_{ij}=\frac{\bm r_{ij}}{r_{ij}}.
\end{equation}
Choose a rotation $F_{ij}\in\SO(3)$ that maps global Cartesian components into
the bond frame and satisfies
\begin{equation}
  F_{ij}\widehat{\bm z}_{ij}=\bm e_z.
  \label{eq:frame-condition}
\end{equation}
The transverse $x$ and $y$ axes are a gauge choice.  Any two valid choices
differ by a rotation around $\bm e_z$.  All bond-frame operations below are
$\SO(2)$ equivariant, so the final global message is independent of this gauge.

For neighbor $k\in\Ncal(ij)$, let
\begin{equation}
  \bm d_{ij,k}=\bm R_k-\overline{\bm R}_{ij},
  \qquad
  \widetilde{\bm d}_{ij,k}=F_{ij}\bm d_{ij,k}
  =
  \begin{pmatrix}
  \rho_{ij,k}\cos\theta_{ij,k}\\
  \rho_{ij,k}\sin\theta_{ij,k}\\
  z_{ij,k}
  \end{pmatrix}.
  \label{eq:cylindrical}
\end{equation}

For periodic systems, the same periodic image convention must be used for the
central bond, its midpoint, and every member of the bond environment.  The
triplet builder should store the required cell shifts explicitly; reconstructing
midpoints from wrapped positions is not safe.

\section{Rotating atomic states into a bond frame}

The Wigner matrix depends on the central bond, not on the environment atom.
For every $(ij,k)\in\Tcal$, rotate the global state of $k$ into the $(i,j)$ bond
frame:
\begin{equation}
  \widetilde h_{k\to ij,c}^{(t,\ell m)}
  =\sum_{m'=-\ell}^{\ell}
  \D_{mm'}^\ell(F_{ij})h_{k,c}^{(t,\ell m')}.
  \label{eq:global-to-local}
\end{equation}
The matrices $\D^\ell(F_{ij})$ are computed once per directed bond and reused
for every $k\in\Ncal(ij)$ and every interaction layer.  Only the contraction
with the changing atomic state is repeated.

The endpoint states may be rotated by the same rule and used to construct a
central-bond conditioner,
\begin{equation}
  b_{ij}^{(t)}=
  \MLP_{\mathrm{bond}}^{(t)}\!\left(
  h_i^{(t,00)},h_j^{(t,00)},P(r_{ij})
  \right),
  \label{eq:bond-conditioner}
\end{equation}
where $P(r_{ij})$ denotes a vector of smooth radial basis functions.  Equation
\eqref{eq:bond-conditioner} does not read element labels directly; chemical
identity is already encoded in $h_i$ and $h_j$.

\section{One-particle cylindrical basis}

The learnable cylindrical radial embedding is an MLP on fixed radial features,
not a separate Bessel index times a longitudinal index.  Its output width is the
narrow density channel count $C_\phi$:
\begin{equation}
  R_{n\mu}^{(t)}(\rho_{ij,k},z_{ij,k})
  =
  \mathrm{MLP}_{\mathrm{env}}^{(t)}
  \!\left(
  \mathrm{Bessel}(\rho_{ij,k}),\,
  Z(z_{ij,k})
  \right)_{n\mu},
  \qquad
  1\le n\le C_\phi,
  \quad |\mu|\le \mu_{\max}.
  \label{eq:radial-mlp}
\end{equation}
Here $Z(z)$ denotes a smooth longitudinal encoding of the signed height in the
bond frame (Bessel features of $|z|$ together with $z/r_{\max}$).
The implementation also concatenates endpoint / environment element one-hots
into the same MLP.  The geometric basis is then
\begin{equation}
  g_{ij,k}^{n\mu}
  =f_{\mathrm{env}}(\rho_{ij,k},z_{ij,k};r_{ij})
  R_{n\mu}^{(t)}(\rho_{ij,k},z_{ij,k})
  \e^{\mathrm{i}\mu\theta_{ij,k}}.
  \label{eq:geometry-basis}
\end{equation}
There is no additional linear map $V$ over old $(n,s)$ indices: that contraction
is absorbed into $\mathrm{MLP}_{\mathrm{env}}$.  Near the bond axis the phase
factor must still behave as $O(\rho^{|\mu|})$, which is automatic if
$\e^{\mathrm{i}\mu\theta}$ is realized through $(\rho\e^{\mathrm{i}\theta})^\mu$.
The cutoff $f_{\mathrm{env}}$ must go smoothly to zero with enough derivatives
for stable force training.

Under a change of transverse gauge by angle $\gamma$,
\begin{equation}
  g^{n\mu}\longmapsto \e^{\mathrm{i}\mu\gamma}g^{n\mu},
  \qquad
  \widetilde h^{\ell m}\longmapsto
  \e^{\mathrm{i}m\gamma}\widetilde h^{\ell m}.
\end{equation}
Therefore their product has $\SO(2)$ order
\begin{equation}
  q=m+\mu.
  \label{eq:so2-selection}
\end{equation}

One-particle features are constructed in the narrow width $C_\phi$, not in the
full node width $C$.  First map the rotated atomic state into density channels.
By default the channel map is block-diagonal in $\ell$,
\begin{equation}
  x_{k\to ij,a}^{(t,\ell m)}
  =\sigma\!\left(b_{ij}^{(t)}\right)_{a}
  \sum_{c=1}^{C}
  U_{a c}^{(t,\ell)}
  \widetilde h_{k\to ij,c}^{(t,\ell m)},
  \qquad
  1\le a\le C_\phi,
  \label{eq:density-projection-u}
\end{equation}
where $\sigma(b_{ij}^{(t)})$ is a learned bond gate shared by all environment
atoms of that bond.  The channelwise specialization
used in the code is $n=a$, so
\begin{equation}
  \phi_{ij,k,a q}^{(t)}
  =
  \sum_{\substack{\ell,m,\mu\\m+\mu=q}}
  C_{a;\ell m\mu}^{(t)}
  x_{k\to ij,a}^{(t,\ell m)}
  g_{ij,k}^{a\mu},
  \qquad
  1\le a\le C_\phi.
  \label{eq:factorized-one-particle}
\end{equation}
With the switch \texttt{mix\_l\_after\_rotation}, the channel map is the linear
family allowed after the bond axis has been aligned with $\bm e_z$.  A residual
rotation by $\gamma$ multiplies $\widetilde h^{\ell m}$ by $\e^{\mathrm{i}m\gamma}$
for every $\ell$, so the map mixes $\ell\ge |m|$ and only $\ell$ of one parity
$p=(-1)^\ell$.  The real matrix depends on $|m|$ and is shared by $+m$ and $-m$:
\begin{equation}
  x_{k\to ij,a}^{(t,m,p)}
  =\sigma\!\left(b_{ij}^{(t)}\right)_{a}
  \sum_{\substack{\ell\ge |m|\\ (-1)^\ell=p}}
  \sum_{c=1}^{C}
  U_{a,\ell c}^{(t,|m|,p)}
  \widetilde h_{k\to ij,c}^{(t,\ell m)}.
  \label{eq:fixed-m-mix}
\end{equation}
Each $\ell$ block of $U^{(t,|m|,p)}$ is initialized to the rectangular identity.
The one-particle feature then has no $\ell$ index,
\begin{equation}
  \phi_{ij,k,a q}^{(t)}
  =
  \sum_{p}
  \sum_{m+\mu=q}
  C_{a;m\mu p}^{(t)}
  x_{k\to ij,a}^{(t,m,p)}
  g_{ij,k}^{a\mu}.
  \label{eq:fixed-m-one-particle}
\end{equation}
The selection rule in Equation \eqref{eq:so2-selection} is the only mandatory
angular coupling rule at this stage.  Constructing $\phi$ at width $C_\phi$ is
what makes the triplet path affordable when $C_\phi<C$.

With \texttt{SO2fulltp}, the channelwise specialization $n=a$ is replaced by a
full bilinear map that depends only on $(|m|,|\mu|)$,
\begin{equation}
  \phi_{ij,k,a q}^{(t)}
  =
  \sum_{m+\mu=q}
  \sum_{b,n=1}^{C_\phi}
  W_{a;bn}^{(t,|m|,|\mu|)}
  x_{k\to ij,b}^{(t,m)}
  g_{ij,k}^{n\mu}.
\end{equation}
With \texttt{SO2lowranktp} and rank $R$, the same map is factorized as
\begin{equation}
  \phi_{ij,k,a q}^{(t)}
  =
  \sum_{m+\mu=q}
  \sum_{r=1}^{R}
  U_{ar}^{(t,|q|)}
  \Bigl(
  \sum_b P_{rb}^{(t,|m|)}x_{k\to ij,b}^{(t,m)}
  \Bigr)
  \Bigl(
  \sum_n Q_{rn}^{(t,|\mu|)}g_{ij,k}^{n\mu}
  \Bigr).
\end{equation}

\section{Bond density projection}

Scatter environment atoms while still in the narrow width,
\begin{equation}
  A_{ij,a q}^{(t)}
  =\sum_{k\in\Ncal(ij)}
  \phi_{ij,k,a q}^{(t)},
  \qquad
  1\le a\le C_\phi,
  \label{eq:density-projection}
\end{equation}
then expand back to the node width with a learned equivariant mix that depends
only on $|q|$:
\begin{equation}
  A'_{ij,\alpha q}^{(t)}
  =
  \sum_{a=1}^{C_\phi}
  O_{\alpha a}^{(|q|)}
  A_{ij,a q}^{(t)},
  \qquad
  1\le \alpha\le C.
  \label{eq:density-expand}
\end{equation}
Because $O$ is linear in the environment sum,
\begin{equation}
  \sum_k O\phi_k = O\sum_k\phi_k,
\end{equation}
moving $O$ after the scatter does not change the represented function class; it
only replaces a length-$|\Tcal|$ GEMM by a length-$|\Ecal|$ GEMM.
The implementation normalizes by the dataset average neighbour count
$\bar n$ rather than by $|\Ncal(ij)|$,
\begin{equation}
  A_{ij,a q}^{(t)}
  \leftarrow
  \frac{A_{ij,a q}^{(t)}}{\bar n}.
  \label{eq:density-normalization}
\end{equation}
Subsequent $\SO(2)$ density products and atomic messages use the expanded
density $A'$.

The order-$q$ transformation law is
\begin{equation}
  A_{ij,a q}^{(t)}
  \longmapsto
  \e^{\mathrm{i}q\gamma}A_{ij,a q}^{(t)}.
\end{equation}

\section{$\SO(2)$ density products}

\subsection{Recursive correlation basis}

Set
\begin{equation}
  B_{ij,\alpha q}^{(t,1)}=A'_{ij,\alpha q}^{(t)}.
\end{equation}
Before every product, apply a learned equivariant channel mixing that is shared
within each real two-component $|q|>0$ block:
\begin{equation}
  \widehat A'_{ij,\alpha q}^{(t,\nu)}
  =\sum_\beta U_{\alpha\beta q}^{(t,\nu)}
  A'_{ij,\beta q}^{(t)}.
\end{equation}
Then form higher correlation orders using an $\SO(2)$ convolution.  The default
implementation is channelwise,
\begin{equation}
  B_{ij,\alpha q}^{(t,\nu+1)}
  =
  \sum_{p+s=q}
  B_{ij,\alpha p}^{(t,\nu)}
  \widehat A'_{ij,\alpha s}^{(t,\nu)},
  \qquad
  1\le\nu<\nu_{\max}.
  \label{eq:so2-convolution}
\end{equation}
With the switch \texttt{full\_self\_tensor\_product}, the channel product is the
full outer product, then a learned map projects $C\times C$ back to width $C$:
\begin{equation}
  T_{ij,\alpha\beta q}^{(t,\nu)}
  =
  \sum_{p+s=q}
  B_{ij,\alpha p}^{(t,\nu)}
  \widehat A'_{ij,\beta s}^{(t,\nu)},
  \qquad
  B_{ij,\gamma q}^{(t,\nu+1)}
  =
  \sum_{\alpha,\beta}
  W_{\gamma\alpha\beta}^{(t,\nu)}
  T_{ij,\alpha\beta q}^{(t,\nu)}.
  \label{eq:so2-full-tp}
\end{equation}
Orders outside $|q|\le q_{\max}$ are discarded.  The full-TP path keeps every
$B^{(\nu)}$ at width $C$; it does not retain an explicit $C^2$ feature
dimension across correlation orders.  At initialization $W$ is diagonal
($W_{\gamma\alpha\beta}=\delta_{\gamma\alpha}\delta_{\alpha\beta}$), so the
full TP starts identical to the channelwise product.

With the switch \texttt{nonlinear\_B}, each $B^{(t,\nu)}$ is passed through an
$\SO(2)$ gated nonlinearity that may mix all channels.  Write
$\widetilde B$ for a learned equivariant channel mix within each $|q|$ block.
The $q=0$ block is invariant, so an MLP may act on it freely:
\begin{equation}
  s_{ij}^{(t,\nu)}
  =
  \mathrm{Re}\,\widetilde B_{ij,\cdot\,0}^{(t,\nu)},
  \qquad
  B_{ij,\alpha 0}^{(t,\nu)}
  \leftarrow
  s_{ij,\alpha}^{(t,\nu)}
  +\mathrm{MLP}_{\mathrm{inv}}^{(t,\nu)}(s_{ij}^{(t,\nu)})_{\alpha}.
\end{equation}
Gates for every nonzero order are also produced from the same invariants,
\begin{equation}
  g_{ij,\alpha |q|}^{(t,\nu)}
  =
  \sigma\!\left(
  \mathrm{MLP}_{\mathrm{gate}}^{(t,\nu)}(s_{ij}^{(t,\nu)})
  \right)_{\alpha |q|},
  \qquad
  B_{ij,\alpha q}^{(t,\nu)}
  \leftarrow
  g_{ij,\alpha |q|}^{(t,\nu)}
  \widetilde B_{ij,\alpha q}^{(t,\nu)}
  \quad(|q|>0).
\end{equation}
Positive and negative $q$ share the gate for $|q|$, preserving
$B_{-q}=B_q^*$.  At initialization the invariant residual and the gate logits
are chosen so that this map starts near the identity.

Expanding Equation \eqref{eq:so2-convolution} shows the density trick:
\begin{equation}
  A^\nu
  =\left(\sum_k\phi_k\right)^\nu
  =\sum_{k_1,\ldots,k_\nu}
  \phi_{k_1}\cdots\phi_{k_\nu}.
\end{equation}
Repeated neighbor indices are included, exactly as in standard ACE/MACE density
products.  Correlation order is therefore not a claim that all participating
atoms are distinct.

\subsection{Invariant and equivariant outputs}

The $q=0$ blocks are invariant under the transverse gauge and can be sent
directly to an energy readout.  To construct an equivariant global atomic
message of angular momentum $L$, assemble a complete local magnetic vector
from matching $\SO(2)$ orders:
\begin{equation}
  \widetilde m_{i\leftarrow j,c}^{(t,L M)}
  =
  \delta_{M0}\,m_{ij,c}^{(t,L,\mathrm{pair})}
  +
  \sum_{\nu=1}^{\nu_{\max}}\sum_\alpha
  W_{c\alpha M}^{(t,L,\nu)}
  \bigl(b_{ij}^{(t)}\bigr)
  B_{ij,\alpha M}^{(t,\nu)},
  \qquad |M|\le L.
  \label{eq:local-message}
\end{equation}
The explicit two-body term has only $M=0$ in the bond frame,
\begin{equation}
  m_{ij,c}^{(t,L,\mathrm{pair})}
  =\sum_q W_{cq}^{(t,L,\mathrm{pair})}
  \bigl(h_i^{(t,00)},h_j^{(t,00)}\bigr)P_q(r_{ij}).
  \label{eq:pair-message}
\end{equation}
After rotation back to the global frame, this term generates the familiar
radial function times a rank-$L$ angular feature along the bond.

\section{Return to the global frame and update atoms}

Because $F_{ij}$ maps global components to local components, its inverse maps
the local message back to the global frame:
\begin{equation}
  m_{i\leftarrow j,c}^{(t,L M)}
  =\sum_{N=-L}^{L}
  \D_{MN}^{L}(F_{ij}^{-1})
  \widetilde m_{i\leftarrow j,c}^{(t,L N)}.
  \label{eq:local-to-global}
\end{equation}
Aggregate incoming directed-bond messages at atom $i$:
\begin{equation}
  M_{i,c}^{(t,L M)}
  =\sum_{j:(i,j)\in\Ecal}
  m_{i\leftarrow j,c}^{(t,L M)}.
  \label{eq:node-scatter}
\end{equation}

For $L>0$, nonlinear activation must not be applied independently to magnetic
components.  Use scalar gates derived from invariant node features:
\begin{align}
  s_i^{(t)}
  &=\left[h_i^{(t,00)},
  \left\{\sum_m
  \left|h_{i,c}^{(t,\ell m)}\right|^2\right\}_{\ell,c}
  \right],\\
  g_{i,c}^{(t,L)}
  &=\sigma\!\left(\MLP_{L}^{(t)}(s_i^{(t)})_c\right),\\
  h_{i,c}^{(t+1,L M)}
  &=h_{i,c}^{(t,L M)}
  +g_{i,c}^{(t,L)}
  \sum_{c'}W_{cc'}^{(t,L)}M_{i,c'}^{(t,L M)}.
  \label{eq:gated-update}
\end{align}
For $L=0$, an ordinary scalar MLP may additionally be used.  Residual updates
are recommended for stable optimization.

\section{Layerwise node and edge energy readouts}

\subsection{Node energy}

Only invariant atomic features may enter the node-energy readout.  The minimal
choice uses the scalar irrep:
\begin{equation}
  \varepsilon_{i,\mathrm{node}}^{(t)}
  =\MLP_{\mathrm{node}}^{(t)}
  \left(h_i^{(t,00)}\right).
  \label{eq:node-energy}
\end{equation}
Norms or scalar tensor-product contractions of higher irreps may be appended,
provided they are true invariants.

\subsection{Directed proposal and physical edge energy}

For directed bond $(i,j)$, collect its invariant bond features,
\begin{equation}
  \xi_{i\to j}^{(t)}=
  \left[
  P(r_{ij}),
  h_i^{(t,00)},h_j^{(t,00)},
  \left\{B_{ij,\alpha 0}^{(t,\nu)}\right\}_{\alpha,\nu}
  \right].
\end{equation}
Define a directed proposal
\begin{equation}
  \widetilde\varepsilon_{i\to j,\mathrm{edge}}^{(t)}
  =\MLP_{\mathrm{edge}}^{(t)}\left(\xi_{i\to j}^{(t)}\right).
\end{equation}
The physical energy of the undirected bond is explicitly reversal symmetric:
\begin{equation}
  \varepsilon_{\{i,j\},\mathrm{edge}}^{(t)}
  =\frac12\left(
  \widetilde\varepsilon_{i\to j,\mathrm{edge}}^{(t)}
  +\widetilde\varepsilon_{j\to i,\mathrm{edge}}^{(t)}
  \right).
  \label{eq:symmetric-edge-energy}
\end{equation}
This is valid for both homoatomic and heteroatomic bonds.  It does not require
the two directed bond environments to be represented identically.

\subsection{Selectable hybrid total energy}

Let $t=0,\ldots,T-1$ index interaction blocks.  Block $t$ constructs the bond
density $B^{(t)}$, produces its edge-energy contribution, and updates
$h^{(t)}$ to $h^{(t+1)}$.  The corresponding node-energy contribution is read
from the updated state $h^{(t+1)}$.  An optional reference or initial-state
readout is denoted by $\varepsilon_{i,\mathrm{node}}^{(0)}$.

Let $\lambda_{\mathrm N}^{(t)}$ and $\lambda_{\mathrm E}^{(t)}$ be fixed switches
or learned scalar coefficients.  A layer-consistent total energy is
\begin{align}
  E={}&
  \lambda_{\mathrm N}^{(0)}
  \sum_i\varepsilon_{i,\mathrm{node}}^{(0)}
  \nonumber\\
  &+\sum_{t=0}^{T-1}
  \left[
  \lambda_{\mathrm N}^{(t+1)}
  \sum_i\varepsilon_{i,\mathrm{node}}^{(t+1)}
  +
  \lambda_{\mathrm E}^{(t)}
  \sum_{\{i,j\}}
  \varepsilon_{\{i,j\},\mathrm{edge}}^{(t)}
  \right].
  \label{eq:total-energy}
\end{align}
The available modes are:
\begin{align}
  \text{node only:}&\quad \lambda_{\mathrm E}^{(t)}=0,\\
  \text{edge only:}&\quad \lambda_{\mathrm N}^{(t)}=0,\\
  \text{hybrid:}&\quad \lambda_{\mathrm N}^{(t)}\ne0,
  \quad\lambda_{\mathrm E}^{(t)}\ne0.
\end{align}
When both readouts are trained only from total energies and forces, their
individual decomposition is not identifiable.  The separate values should not
be interpreted as unique physical atomic and bond energies without additional
constraints or supervision.

As in MACE, layerwise readouts provide short gradient paths.  Early layers can
use linear readouts, while the final layer may use an MLP.

\section{Energy derivatives}

Forces are obtained by automatic differentiation,
\begin{equation}
  \bm F_i=-\frac{\partial E}{\partial\bm R_i}.
\end{equation}
All operations that depend on geometry, including midpoint construction,
cylindrical coordinates, Wigner rotations, cutoffs, radial bases, scatter sums,
and energy readouts, must remain in the differentiable computation graph.

For a periodic cell matrix $H$, a consistent strain derivative gives the
stress.  One common convention is
\begin{equation}
  \bm\sigma=\frac{1}{\Omega}
  \frac{\partial E}{\partial\bm\epsilon}
  \bigg|_{\bm\epsilon=0},
  \qquad H\mapsto(I+\bm\epsilon)H,
\end{equation}
with the sign adjusted to match the dataset and MACE convention.

\section{Real-valued implementation}\label{sec:real}

For $q>0$, store the complex coefficient $z_q=a_q+\mathrm{i}b_q$ as the real
two-vector
\begin{equation}
  [a_q,b_q]=[\RePart z_q,\ImPart z_q].
\end{equation}
For a real density, negative orders obey $z_{-q}=z_q^*$.  It is therefore
sufficient to store $q\ge0$.

If $z_p=a+\mathrm{i}b$ and $z_q=c+\mathrm{i}d$, the sum-order product is
\begin{equation}
  z_pz_q=(ac-bd)+\mathrm{i}(ad+bc),
\end{equation}
and the difference-order product is
\begin{equation}
  z_pz_q^*=(ac+bd)+\mathrm{i}(bc-ad).
\end{equation}
An order-zero factor is an ordinary scalar multiplication.  Learned channel
mixing must act identically on the cosine and sine components of a fixed
$q>0$ block.  These rules implement Equation \eqref{eq:so2-convolution} without
complex tensors.

For the expected range $q_{\max}\le4$, a direct sparse convolution is usually
preferable to an FFT.  FFT or nonuniform FFT implementations should be
considered only after profiling substantially larger angular bandwidths.

\section{Equivariance argument}

Let a global rotation $Q$ act on all positions and atomic states.  Both
$F_{ij}(\bm R)$ and $F_{ij}(Q\bm R)Q$ map the original bond axis to $\bm e_z$.
Consequently, they differ by a residual bond-axis rotation
$R_z(\gamma_{ij})$:
\begin{equation}
  F_{ij}(Q\bm R)Q
  =R_z(\gamma_{ij})F_{ij}(\bm R).
  \label{eq:gauge-relation}
\end{equation}
Equations \eqref{eq:global-to-local}--\eqref{eq:so2-convolution} ensure that a
local order-$q$ feature acquires exactly the phase
$\exp(\mathrm{i}q\gamma_{ij})$.  Equation \eqref{eq:local-message} selects order
$M$ for the local magnetic component $M$.  Rotating back with
$\D^L(F_{ij}^{-1})$ cancels the gauge rotation and yields
\begin{equation}
  m_{i\leftarrow j}^{(t,L)}
  \longmapsto
  \D^L(Q)m_{i\leftarrow j}^{(t,L)}.
\end{equation}
Scatter sums and gated updates preserve the same transformation law.  Scalar
readouts are therefore invariant, and the energy in Equation
\eqref{eq:total-energy} is invariant under global rotations.

Reflection symmetry is an additional modeling decision.  For ordinary
nonrelativistic potential energies, one normally targets $\OO(3)$ rather than
only $\SO(3)$.  In a real implementation, parity labels must be assigned to
global irreps and to the longitudinal/transverse basis so that only
parity-allowed products are retained.  Taking a real part alone guarantees a
particular transverse reflection symmetry but does not by itself prove full
$\OO(3)$ equivariance for heteroatomic directed bonds.

\section{Computational complexity and efficient execution}

Let $N$ be the number of atoms, $E$ the number of directed central bonds,
$K$ the average size of a bond environment, and $P\approx EK$ the number of
stored bond-environment triplets.  Let $C$ denote a representative channel
count and $M=2q_{\max}+1$ the number of signed Fourier modes.

\begin{enumerate}[leftmargin=2em]
  \item Build $F_{ij}$, $\D^\ell(F_{ij})$, and central radial bases once per
  directed bond: $O(E)$ geometric objects.
  \item Build cylindrical geometry bases once per triplet: $O(P)$ objects.
  \item At each layer, gather atomic states and rotate them with the Wigner
  matrices indexed by the central bond.  The matrices are stored once per bond,
  although contractions occur once per triplet.
  \item Form one-particle features and scatter them to bonds: approximately
  $O(PCM)$ for a factorized product.
  \item Form correlation products: approximately
  $O(E\nu_{\max}CM^2)$ for direct convolution.  The small, sparse selection
  rules make the practical cost lower than the dense bound.
  \item Rotate local messages back once per directed bond and scatter to atoms.
\end{enumerate}

The crucial saving is the absence of an $O(EK^\nu)$ explicit neighbor-tuple
enumeration.  Additional implementation rules are:
\begin{itemize}[leftmargin=2em]
  \item pre-mix to a smaller product-channel dimension before multiplying;
  \item use channelwise products followed by output mixing;
  \item fuse gather, rotation, basis product, and scatter when practical;
  \item never materialize a dense tensor over all $(m,\mu,q)$ combinations;
  store only allowed index triples satisfying $q=m+\mu$;
  \item compute both directed energies but store one symmetric undirected
  edge-energy contribution;
  \item start with small $\ell_{\max}$, $q_{\max}$, and $\nu_{\max}$ and profile
  before increasing angular bandwidth.
\end{itemize}

\section{MACE-oriented module decomposition}

A practical code organization is:
\begin{enumerate}[leftmargin=2em]
  \item \texttt{ElementEmbedding}: Equation \eqref{eq:element-init}.
  \item \texttt{BondEnvironmentBuilder}: edges, midpoints, periodic shifts, and
  the triplet list $\Tcal$.
  \item \texttt{BondFrame}: $F_{ij}$ and real Wigner matrices for every $\ell$.
  \item \texttt{CylindricalBasis}: Equation \eqref{eq:geometry-basis}.
  \item \texttt{GlobalToBondRotation}: Equation
  \eqref{eq:global-to-local}.
  \item \texttt{BondDensity}: factorized one-particle features and the scatter
  in Equation \eqref{eq:density-projection}.
  \item \texttt{SO2DensityProduct}: Equation \eqref{eq:so2-convolution} in real
  cosine/sine blocks.
  \item \texttt{BondMessage}: Equations \eqref{eq:local-message} and
  \eqref{eq:pair-message}.
  \item \texttt{BondToGlobalRotation}: Equation
  \eqref{eq:local-to-global}.
  \item \texttt{EquivariantNodeUpdate}: Equations \eqref{eq:node-scatter} and
  \eqref{eq:gated-update}.
  \item \texttt{NodeReadout} and \texttt{EdgeReadout}: Equations
  \eqref{eq:node-energy}--\eqref{eq:total-energy}.
\end{enumerate}

MACE already provides batching, neighbor graphs, irreps bookkeeping, radial
bases, scatter operations, and energy/force training infrastructure.  The new
components are primarily the midpoint-centered triplet builder, bond-frame
rotation, cylindrical basis, $\SO(2)$ density product, and the symmetric edge
readout.

\section{Reference forward pass}

The following pseudocode makes tensor ownership and reuse explicit.

\begin{lstlisting}[language=Python]
# Static geometry for one forward pass
edges = build_directed_edges(positions, cell, edge_cutoff)
triplets = build_bond_environment_triplets(
    edges, positions, cell, environment_cutoff, exclude_endpoints=True
)
F = build_bond_frames(edges, positions, cell)              # [E, 3, 3]
D = {l: real_wigner_D(F, l) for l in range(l_max + 1)}     # per bond
geom = cylindrical_basis(edges, triplets, positions, cell) # per triplet
P = central_bond_radial_basis(edges, positions, cell)      # per bond

# Element identity enters only here.
h = initialize_irreps_to_zero(number_of_atoms, irreps_hidden)
h[0] = element_embedding(atomic_numbers)  # l = 0 block

total_energy = 0.0
if use_initial_node_readout:
    total_energy += node_readout[0](invariants_of_nodes(h)).sum()

for t in range(number_of_layers):

    # Gather h_k for every (ij, k), then rotate with D(F_ij).
    h_local = rotate_gathered_atoms_to_bonds(h, D, triplets)

    # Factorized h_local times cylindrical geometry, grouped by q = m + mu.
    phi = one_particle_product[t](h_local, geom, P, edges, triplets)
    A = scatter_sum(phi, triplets.center_bond, dim_size=number_of_edges)

    # B[nu] is built from the aggregated density, never from neighbor tuples.
    B = [A]
    for nu in range(1, nu_max):
        B.append(so2_channelwise_convolution(B[-1], mix[t][nu](A)))

    if use_edge_readout:
        directed = edge_readout[t](P, scalar_endpoints(h, edges), q0(B))
        undirected = symmetrize_reverse_edges(directed, edges.reverse_index)
        total_energy += sum_each_undirected_edge_once(undirected)

    m_local = make_local_irrep_message[t](P, scalar_endpoints(h, edges), B)
    m_global = rotate_bond_messages_to_global(m_local, D)
    aggregated = scatter_sum(m_global, edges.sender, dim_size=number_of_atoms)
    h = equivariant_residual_update[t](h, aggregated)

    if use_node_readout:
        total_energy += node_readout[t + 1](invariants_of_nodes(h)).sum()

forces = -grad(total_energy, positions)
\end{lstlisting}

In a MACE-style implementation, one may also evaluate the edge readout after
the node update.  The convention must be chosen once and used consistently in
the definition of the layer index.

\section{Recommended first implementation}

A conservative first model is
\begin{equation}
  \ell_{\max}=2,
  \qquad
  q_{\max}=2,
  \qquad
  \nu_{\max}=2,
  \qquad
  T=2.
\end{equation}
Suggested initial dimensions are $32$--$64$ persistent channels per selected
global irrep and $16$--$32$ bond-product channels.  Use direct real $\SO(2)$
products, a smooth midpoint-centered cutoff, an explicit pair path, residual
gated updates, and layerwise linear readouts except for the final readout.

The implementation should be validated in the following order:
\begin{enumerate}[leftmargin=2em]
  \item translation invariance and permutation invariance;
  \item energy invariance under random global rotations;
  \item equivariance of every $h^{(\ell)}$ block and of forces;
  \item independence of the transverse bond-frame gauge;
  \item reversal symmetry of the edge-energy readout;
  \item finite-difference agreement of forces;
  \item equality between explicit low-order neighbor-tuple sums and the density
  trick on tiny test systems;
  \item periodic image consistency near cell boundaries.
\end{enumerate}

Only after these tests pass should one compare node-only, edge-only, and hybrid
readouts, or increase $\nu_{\max}$ and angular bandwidth.

\section{Summary of the complete computation}

For quick reference, one interaction layer consists of:
\begin{align}
  \widetilde h_{k\to ij}^{(t,\ell m)}
  &=\sum_{m'}\D_{mm'}^\ell(F_{ij})h_k^{(t,\ell m')},\\
  x_{k\to ij,a}^{(t,\ell m)}
  &=\sigma(b_{ij}^{(t)})_a\sum_c U_{ac}^{(t,\ell)}
  \widetilde h_{k\to ij,c}^{(t,\ell m)},\\
  R_{a\mu}^{(t)}
  &=\mathrm{MLP}_{\mathrm{env}}^{(t)}
  \!\bigl(\mathrm{Bessel}(\rho),Z(z)\bigr)_{a\mu},\\
  g_{ij,k}^{a\mu}
  &=f_{\mathrm{env}}R_{a\mu}^{(t)}
  \e^{\mathrm{i}\mu\theta},\\
  \phi_{ij,k,a q}^{(t)}
  &=\sum_{m+\mu=q}C_{a;\ell m\mu}^{(t)}
  x_{k\to ij,a}^{(t,\ell m)}g_{ij,k}^{a\mu},\\
  A_{ij,a q}^{(t)}
  &=\sum_{k\in\Ncal(ij)}\phi_{ij,k,a q}^{(t)},\\
  A'_{ij,\alpha q}^{(t)}
  &=\sum_a O_{\alpha a}^{(|q|)}A_{ij,a q}^{(t)},\\
  B_{ij,\alpha q}^{(t,1)}&=A'_{ij,\alpha q}^{(t)},\\
  B_{ij,\alpha q}^{(t,\nu+1)}
  &=\sum_{p+s=q}B_{ij,\alpha p}^{(t,\nu)}\widehat A'_{ij,\alpha s}^{(t)},\\
  \widetilde m_{i\leftarrow j}^{(t,LM)}
  &=\delta_{M0}m_{ij}^{(t,L,\mathrm{pair})}
  +\sum_{\nu,\alpha}W_{\alpha M}^{(t,L,\nu)}B_{ij,\alpha M}^{(t,\nu)},\\
  m_{i\leftarrow j}^{(t,LM)}
  &=\sum_N\D_{MN}^{L}(F_{ij}^{-1})
  \widetilde m_{i\leftarrow j}^{(t,LN)},\\
  M_i^{(t,LM)}&=\sum_jm_{i\leftarrow j}^{(t,LM)},\\
  h_i^{(t+1,LM)}&=h_i^{(t,LM)}+
  g_i^{(t,L)}W^{(t,L)}M_i^{(t,LM)}.
\end{align}
Interaction block $t$ can then contribute
\begin{equation}
  E^{(t)}=
  \lambda_{\mathrm N}^{(t+1)}
  \sum_i\varepsilon_{i,\mathrm{node}}^{(t+1)}
  +\lambda_{\mathrm E}^{(t)}\sum_{\{i,j\}}
  \varepsilon_{\{i,j\},\mathrm{edge}}^{(t)},
\end{equation}
and the full energy is
\begin{equation}
  E=\lambda_{\mathrm N}^{(0)}
  \sum_i\varepsilon_{i,\mathrm{node}}^{(0)}
  +\sum_{t=0}^{T-1}E^{(t)}.
\end{equation}

\end{document}
