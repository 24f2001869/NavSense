# Methodology & Mathematical Formulations

This document provides the formal mathematical derivations governing the SIH26168 navigation engine.

---

## 1. 3D Inertial Strapdown Mechanization

The attitude quaternion $\mathbf{q}_b^n$ is integrated using zero-order hold over interval $\Delta t$:

$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \begin{bmatrix} \cos(\|\boldsymbol{\theta}_k\| / 2) \\ \frac{\boldsymbol{\theta}_k}{\|\boldsymbol{\theta}_k\|} \sin(\|\boldsymbol{\theta}_k\| / 2) \end{bmatrix}$$

where $\boldsymbol{\theta}_k = (\boldsymbol{\omega}_k - \hat{\mathbf{b}}_g) \Delta t$.

Specific force is rotated to navigation frame and integrated:

$$\mathbf{f}_k^n = \mathbf{R}(\mathbf{q}_k) (\mathbf{a}_k - \hat{\mathbf{b}}_a)$$
$$\mathbf{v}_{k+1}^n = \mathbf{v}_k^n + (\mathbf{f}_k^n + \mathbf{g}^n) \Delta t$$
$$\mathbf{p}_{k+1}^n = \mathbf{p}_k^n + \frac{\mathbf{v}_k^n + \mathbf{v}_{k+1}^n}{2} \Delta t$$

---

## 2. 15-State Error-State Kalman Filter (ESKF)

The state transition matrix $\boldsymbol{\Phi}_k = \exp(\mathbf{F} \Delta t)$ is approximated by first-order expansion:

$$\boldsymbol{\Phi}_k \approx \mathbf{I}_{15} + \mathbf{F}_k \Delta t$$

where the continuous error dynamics Jacobian $\mathbf{F}_k$ is:

$$\mathbf{F}_k = \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -\lfloor \mathbf{f}_k^n \times \rfloor & -\mathbf{R}_b^n & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -\lfloor \boldsymbol{\omega}_k^n \times \rfloor & \mathbf{0}_{3 \times 3} & -\mathbf{R}_b^n \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3}
\end{bmatrix}$$

---

## 3. Dilated Temporal Convolutional Network (TCN)

Let $\mathbf{x} \in \mathbb{R}^{T \times C}$ be the input sequence with $C=9$ features over $T=100$ steps. A 1D dilated causal convolution operation $*_d$ on sequence element $s$ is defined as:

$$y(s) = (\mathbf{x} *_d \mathbf{f})(s) = \sum_{i=0}^{K-1} \mathbf{f}(i) \cdot \mathbf{x}_{s - d \cdot i}$$

where $\mathbf{f}$ is a filter of size $K=3$ and $d$ is the dilation factor. The receptive field $\mathcal{R}$ of a network with $L$ residual blocks is:

$$\mathcal{R} = 1 + \sum_{l=1}^L (K_l - 1) \cdot d_l$$

For our 6-layer architecture with $d \in \{1, 2, 4, 8, 16, 32\}$ and $K=3$:

$$\mathcal{R} = 1 + 2 \cdot (1 + 2 + 4 + 8 + 16 + 32) = 1 + 2 \cdot 63 = 127 \text{ steps}$$

which fully encompasses the 100-step (10.0 second) input window.
