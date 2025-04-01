# Holographic Phase Retrieval of RF-EMFs for Human Exposure Assessment using Fine 3D Electric Field Measurements in a Real Environment from a Beamforming 28 GHz Antenna

The increasing use of 5G and 6G technologies necessitates accurate RF-EMF exposure assessment. Both measurements and simulations are used, but matching their results in real environments is challenging due to phase determination issues and the need for precise digital twins. This paper introduces a novel holographic phase retrieval method to reconstruct the electric field from phaseless measurements, validated by measurements in a real environment using a 28 GHz antenna and a positioning robot.

## Methods

### Measurement setup

The measurement setup is shown in Fig. 1. The environment is a rectangular room with smooth concrete walls, a door, and a U-shaped stairwell. The measurement takes place to the left of the door, below the second staircase. The $y$-axis is parallel to the door, pointing from the wall to the staircase. The $z$-axis points up from the floor and the $x$-axis is normal to the door, pointing out of the room.

An antenna is mounted on a pole, radiating at 28 GHz at a height of 1.68 m tilting downward with an angle of 11.5°. The antenna, as detailed in [7], consists of 4 antenna elements and a passive Butler Matrix. Here, the first three ports are terminated and the fourth is fed a sinusoidal signal, such that a directed beam is cast toward the measurement setup at an angle of -14° [8].

The measurement is performed similarly as in [5] and first introduced in [9]. A calibrated uniaxial EMF probe measures the magnitude of the longitudinal electric field component. We use the Narda 5G FR2 probe with a bandwidth 26.5 GHz to 29.5 GHz connected to the Narda SRM-3006 [10], which in turn is connected to a computer. The probe is mounted on a 3D positioning robot, consisting of 3 Velmex slides in orthogonal directions. The actuators move with sub-mm precision. The robot and the probe are positioned and mounted to align at 90° angles with the walls of the room. The longitudinal probe direction is therefore aligned with the $y$-axis.

The SRM-3006 and 3D positioning robot are operated synchronously by a laptop. The SRM-3006, laptop, and all other equipment are placed below the first staircase to minimize the error with the digital twin. The robot traces out a snake-like path covering a 3D domain, while the probe measures continually. The time series of measured field values is mapped to points along the traced path. This approach increases the efficiency substantially compared to [5] and changes the bottleneck of the measurement's duration to the speed of the actuators instead of the probe measurements. Moreover, the resolution is continuous (only limited by the measurement time-interval) along one axis of the snake-like path but discrete along the other two, further referred to as the *discrete-axis resolution*. The 3D domain have a size of 85 cm × 40 cm × 70 cm. For a discrete-axis resolution of λ/4, it takes 31 hours to measure the 5 slices at x = 0 cm, x = 40 cm, y = 0 cm, y = 20 cm and y = 40 cm.

A digital twin of the environment was carefully constructed. The architectural model of the building was provided to us in Revit and converted to OpenUSD. We verified that the architectural model's distances conformed to the distance measurements in the room. The position of the antenna, the robot, and the probe was measured and located in the digital twin.

### Holographic Phase Retrieval method

To compute the theoretical exposure on a human of the antenna, the EMFs $\mathbf{e}(\mathbf{r})$ and $\mathbf{h}(\mathbf{r})$ need to be known in the full 3D domain. However, the measurement only provides $|e_y(\mathbf{r})|$ on certain slices of the 3D domain. We propose an algorithm that reconstructs the full EMF which match the measured $|e_y(\mathbf{r})|$.

According to the Huygens-Fresnel principle, every wavefront is the sum of a number of spherical wavelets emanating from points in the environment. From Maxwell's equations, the electric field can be expressed as [11]

$\mathbf{e}(\mathbf{r}) = -j\omega \mu_0 \int_S \overline{\overline{\mathbf{G}}}(\mathbf{r}, \mathbf{r}') \cdot \mathbf{j}(\mathbf{r}') \, \mathrm{d}\mathbf{r}' \, ,$

where $\mathbf{j}$ is the current density on the physical features of the environment $S$. Green's tensor $\overline{\overline{\mathbf{G}}}$ is explicitly given by

$\overline{\overline{\mathbf{G}}}(\mathbf{r},\mathbf{r}') = \left( \mathbf{I} + \frac{1}{k^2} \boldsymbol{\nabla\nabla} \right) G(R) \, ,$

where $R=\lVert\mathbf{R}\rVert = \left|\mathbf{r} - \mathbf{r}'\right|$ and $G(R)=-\exp(jkR)/R$. One can prove in the far-field limit that all non-radiative components become negligible and

$\lim_{kR\to\infty} \overline{\overline{\mathbf{G}}}(\mathbf{r},\mathbf{r}') = \frac{\exp ( -j k\left|\mathbf{r} - \mathbf{r}'\right|)}{4\pi\left|\mathbf{r} - \mathbf{r}'\right|} \left( \mathbf{I} - \hat{\mathbf{R}}\hat{\mathbf{R}}\right) \, ,$

where $\hat{\mathbf{R}} = \mathbf{R}/\left|\mathbf{R}\right|$.
The term $\mathbf{I} - \hat{\mathbf{R}}\hat{\mathbf{R}}$ projects the current density onto the plane perpendicular to $\hat{\mathbf{R}}$. The integral can be approximated by a sum over a large number of *scattering clusters* $C$, assuming $\mathbf{r}$ is in the far-field of each cluster, as:

$\mathbf{e}(\mathbf{r}) \approx \sum_{c=0}^C \left( a^\theta_c \hat{\mathbf{u}}^\theta_c(\mathbf{r}) + a^\phi_c \hat{\mathbf{u}}^\phi_c(\mathbf{r}) \right) \frac{\exp ( -j k\left|\mathbf{r} - \mathbf{r}_c\right|)}{\left|\mathbf{r} - \mathbf{r}_c\right|} \, ,$

where $\theta$ and $\phi$ denote the polarizations along the spherical unit vectors, $a_c^{\theta / \phi}$ represents the $\theta$ and $\phi$ *coefficients* of cluster $c$, $\hat{\mathbf{u}}_c(\mathbf{r})$ is the polarization unit vector of the vector connecting the cluster at $\mathbf{r}_c$ with $\mathbf{r}$. For an infinite amount of clusters padding the surface $S$ completely, the coefficients of a specific cluster $c$ represent the amplitude and phase of the current density $-j\omega \mu_0 \mathbf{j}(\mathbf{r'})$, projected onto the polarization axes $\hat{\mathbf{u}}_c^{\theta/\phi}$ of $c$.

Finding the EMF entails finding the coefficients of all the clusters $a_c^{\theta / \phi}$. The vectors $\mathbf{a}^\theta$ and $\mathbf{a}^\phi$ collect all these coefficients for the $\theta$ and $\phi$ polarizations, respectively. We consider $K$ points $\mathbf{r}_i$ where measurements of the electric field are known. As only the $y$-component is measured, we premultiply the equation above by $\hat{\mathbf{u}}_y$ and write this in matrix notation

$$\hat{\mathbf{u}}_y \cdot \begin{bmatrix}
\mathbf{e}(\mathbf{r}_1) \\
\vdots \\
\mathbf{e}(\mathbf{r}_K)
\end{bmatrix}
=
\hat{\mathbf{u}}_y \cdot
\begin{bmatrix}
\overline{\overline{\mathbf{H}}}^\theta & \\
& \overline{\overline{\mathbf{H}}}^\phi
\end{bmatrix}
\begin{bmatrix}
\mathbf{a}^\theta \\
\mathbf{a}^\phi
\end{bmatrix} \, .$$

where the elements of the $\overline{\overline{\mathbf{H}}}^{\theta/\phi}$ tensors are

$h_{ic}^{\theta/\phi} = \frac{\exp ( -j k\left|\mathbf{r}_i - \mathbf{r}_c\right|)}{\left|\mathbf{r}_i - \mathbf{r}_c\right|} e_c^{\theta/\phi}(\mathbf{r}_i) \, .$

This equation can be solved for the cluster coefficients in a least-squares sense with the Moore-Penrose pseudoinverse [12, 13]:

$\mathbf{x} = \mathbf{H}^{+} \mathbf{y} =
\begin{bmatrix}
\mathbf{V}_\theta\mathbf{\Sigma_\theta}^{+}\mathbf{U}_\theta^H & \\
& \mathbf{V}_\phi\mathbf{\Sigma_\phi}^{+}\mathbf{U}_\phi^H
\end{bmatrix}
\mathbf{y} \, ,$

where the singular value decomposition of the channel matrix $\mathbf{H}^{\theta / \phi} \in \mathbb{C}^{K\times C}$ is used.

Moreover, only the magnitudes $\mathbf{y}^\mathrm{meas} = \left|\mathbf{y}\right|$ of the $y$-components are known. The equations above are the spatial forward and inverse Fourier transforms between representations in the *cluster domain* and *field domain*. We propose a holographic phase retrieval method that computes an electromagnetically consistent set of field values whose amplitudes converge to the measured values. This is based on the Gerchberg–Saxton algorithm [6] and is described in Algorithm 1. The resulting cluster coefficients are used in the equation above to compute $\mathbf{e}(\mathbf{r})$ (and trivially $\mathbf{h}(\mathbf{r})$) everywhere in the 3D domain.

**Algorithm 1: Holographic Phase Retrieval with the Gerchberg–Saxton Algorithm**

1.  **Input:** Channel matrix $\mathbf{H}$, measured fields $\mathbf{y}_\mathrm{meas}$, convergence threshold $\varepsilon_\mathrm{conv}$
2.  **Initialize:** $n \gets 0$, $\mathbf{x}^{(1)} \gets \mathbf{H}^{+}  \mathbf{y}_\mathrm{meas}$ (Measured values have no phase)
3.  **While** $\varepsilon^{(n)} \geq \varepsilon_\mathrm{conv}$ **do**
4.  $\mathbf{y}_\mathrm{sim}^{(n)} \gets \mathbf{H} \mathbf{x}^{(n)}$ (Forward Fourier Transform)
5.  **For** $i = 1, \dots, K$ **do**
6.  $y_i^{(n)} \gets \left| y_{\mathrm{meas}} \right| y_{i,\mathrm{sim}}^{(n)} / \left| y_{i,\mathrm{sim}}^{(n)} \right|$
7.  **End For**
8.  $\mathbf{x}^{(n+1)} \gets \mathbf{H}^{+} \mathbf{y}^{(n)}$ (Inverse Fourier Transform)
9.  $\varepsilon^{(n)} \gets \left\lVert \mathbf{y}^{(n)} - \mathbf{y}_\mathrm{meas} \right\rVert / \left\lVert \mathbf{y}_\mathrm{meas} \right\rVert$
10. $n \gets n + 1$
11. **End While**
12. **Output:** $\mathbf{x}^{(n)}$
