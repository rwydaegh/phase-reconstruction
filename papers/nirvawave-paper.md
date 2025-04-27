# NirvaWave: An Accurate and Efficient Near Field Wave Propagation Simulator for 6G and Beyond

## Introduction

Millimeter-wave (mmWave) and sub-Terahertz (sub-THz) communications offer high data rates but suffer from severe propagation loss, necessitating directional antennas. The resulting large array apertures extend the near-field region, rendering existing channel simulators inaccurate or computationally expensive. Accurate EM solvers lack scalability, while less demanding simulators neglect spherical wavefronts, making them unsuitable for near-field analysis. Therefore, a precise and efficient near-field channel simulator is crucial for advancing mmWave and sub-THz communications.

## Near-Field Channel Modeling and Simulator Design

### Primer on Rayleigh-Sommerfeld Integral Theory

Assume we have a transmitter array at $x=x_0$, as shown in Fig. 1. The Rayleigh-Sommerfeld theory suggests that we can consider this aperture as an infinite number of point sources each emitting spherical waves. Further, to find the electric field in an arbitrary point (x,y,z), we have to add the contributions from all these point sources together. Since in this theory, there is no assumption about the initial E-field distribution, it allows us to implement any phase/amplitude profile on TX antenna arrays and find the corresponding electric field at arbitrary RX locations or observation planes. The general form of the Rayleigh-Sommerfeld integral can be derived from Maxwell's equations by use of Green's Theorem as follows:

$$E(x', y', z'|E_0) = \frac{1}{2\pi} \iint_A E_0(x_0, y, z) \frac{\Delta x e^{-ikr}}{r^2} \left(ik + \frac{1}{r}\right) dy dz$$

where, $E(x_0, y, z)$ represents the initial E-field distribution at $x_0$. This integral theory would find the complex E-field at any arbitrary point $(x', y', z')$. $k$ represents the wave number, $\Delta x = x' - x_0$, and $r = \sqrt{(x' - x_0)^2 + (y' - y)^2 + (z' - z)^2}$ is the distance from each source point to the desired location.

Although the Rayleigh-Sommerfeld integral provides a straightforward method to find the EM wave evolution as it propagates between TX and RX, it requires computing a discrete integral at each point in space, which is significantly time-consuming. The Angular Spectrum Method (ASM) simplifies the computation of this integral by transforming the equation into the frequency domain using Fourier principles, leveraging the fact that the equation can be written in the form of convolution. Particularly, using ASM, we can write:

$$\mathcal{F}\{E(x', y', z'|E_0)\} = \mathcal{F}\{E(x_0, y, z)\} \times \mathcal{F}\left\{\frac{1}{2\pi} \frac{\partial}{\partial x'} \frac{e^{-ikr}}{r}\right\}$$

This way we can interpret the free space propagation as a linear system in which $\mathcal{F}\left\{\frac{1}{2\pi} \frac{\partial}{\partial x'} \frac{e^{-ikr}}{r}\right\}$ can be viewed as a transfer function that captures the EM disturbance of a point source transmission. In other words, we can write:

$$H(f_y, f_z) = \mathcal{F}\left\{\frac{1}{2\pi} \frac{\partial}{\partial x'} \frac{e^{-ikr}}{r}\right\} = \exp\left(-i2\pi \frac{(x' - x)}{\lambda} \sqrt{1 - \lambda^2(f_y^2 + f_z^2)}\right)$$

where $\lambda$ is the wavelength, $f_y$ and $f_z$ represent spatial frequencies. We highlight that the transfer function $H(f_y, f_z)$ is solely a function of the geometry of the environment and not the transmitted electric field. Finally, $E(x', y', z'|E_0)$ can be calculated by taking a Fourier inverse as follows:

$$E(x', y', z'|E_0) = \mathcal{F}^{-1}\{\mathcal{F}\{E(x_0, y, z)\} \times H(f_y, f_z)\}$$

In NirvaWave, we use the simplified 2D version of these equations. Indeed, Fourier transforms are much more computationally efficient and help reduce the complexity and run-time of simulations in NirvaWave. However, such calculation only applies to free-space communication and other techniques needed to extend it to a wireless medium that involves blockers and reflectors.

### Near-Field Blockage Modeling and Simulation

The Rayleigh-Sommerfeld integral theory originally describes EM wave propagation in free space. However, to account for diffraction due to environmental blockages, we must model the relevant boundary conditions. In free space, using the angular spectrum method, we can compute the electric field at each location $x = x'$ either based on the initial E-field at $x = x_0$ in one shot or based on the previously calculated E-field at $x = x' - \delta x$ using an iterative approach. However, when there are blockages in the environment, we must rely on an iterative scheme to account for any disturbances and discontinuities in the E-field caused by the presence of blockers.

$\delta x$ represents the discrete step size for iterative calculations, and $R_L$ and $T$ denote blocker length and thickness respectively. In NirvaWave, we use $BL(x, y)$ to characterize arbitrary-shaped blockers. Specifically, if $BL(x, y) = 1$, the signal is unattenuated at point $(x, y)$, indicating no blocker. Conversely, $BL(x, y) = \alpha$, where $0 \leq \alpha < 1$, captures the attenuation constant caused by a blocker at (x,y). Thus, the E-field distribution at the observation plane can be determined through an iterative process with the $k$th iteration for $x = x_0 + k\delta x$ expressed as:

$$E(x, y) = BL(x, y) \times \mathcal{F}^{-1}\{H(f_y) \times \mathcal{F}\{E(x-\delta x, y)\}\}$$

Using this approach, NirvaWave is able to account for the diffraction behavior of EM wave propagation in the presence of blockages in a time-efficient manner.

### Near-Field Reflection Modeling and Simulation

For simplicity and without loss of generality, we will first explain how to characterize a single near-field reflector in the environment, followed by a discussion of the general case using a recursive algorithm. To this end, NirvaWave starts by calculating the E-field in the environment assuming there are no reflections. Then, we can find the electric field incident on the reflector's surface denoted as $E(x_{ref}, y_{ref})$, where $(x_{ref}, y_{ref})$ represents the points on the reflector. Building on top of the Huygens-Fresnel principle, we treat the reflector as another source of EM signals in the medium, determined using the previously calculated initial radiating E-field. In other words, similar to modeling the transmitter, we consider the reflector as an infinite number of point sources radiating spherical wavefronts. Therefore, the reflected signal can also be characterized by the Angular Spectrum Method.

It is important to note that ASM originally describes EM wave propagation when the direction of propagation is normal to the E-field source plane. However, when considering reflections, the propagation direction is no longer normal to the reflector plane. Accordingly, the original transfer function must be modified. Indeed, the rotation angle between reflector and virtual source planes (denoted as $\theta$) is purely a function of reflector orientation relative to TX. Hence, the transfer function can be expressed based on full diffraction theory as:

$$H_m(f_y|\theta) = \exp\left(-i2\pi \frac{(x' - x)}{\lambda\cos(\theta)} \sqrt{1 - (\lambda f_y \cos(\theta))^2}\right)$$

Therefore, the EM wave propagation of the reflected wave can be derived using the equation by considering the modified transfer function $H_m(f_y|\theta)$ and the virtual initial EM wave source $E_{vir}(x_0, y) = E(x_{ref}, y_{ref})$. Hence, we can compute the reflected E-field profile in the reflector coordinate system through an iterative process:

$$E_{vir}(x, y) = BL_{ref}(x, y) \times \mathcal{F}^{-1}\{H_m(f_y|\theta) \times \mathcal{F}\{\Gamma_r \times E_{vir}(x-\delta x, y)\}\}$$

where $BL_{ref}(x, y)$ denotes the blocker properties in the reflector coordinate system and $\Gamma_r$ is the reflection coefficient (between 0-1) that can be input by the users. Using this approach, we can model the reflected EM wave propagation in the coordinates of the corresponding reflector plane.

### Diffuse Rough Scattering Implementation in NirvaWave

In practice, reflection at high frequencies also includes diffuse scattering components as the small perturbations on the surface of the reflection become comparable with the sub-mm wavelength of the impinging waves. Indeed, past work showed such scattering behavior has important implications for signal coverage and mobility resilience in sub-THz wireless networks.

Surface perturbations are often modeled with a Gaussian distribution, with height at location $(x)$ as $H(x) \sim \mathcal{N}(0, h_{rms}^2)$, where $h_{rms}$ is the standard deviation of the surface height. Similarly, we can define correlation length $L_c$ as an indicator of horizontal roughness. NirvaWave allows users input $(h_{rms}, L_c)$ parameters that capture the statistical profile of random diffuse scattering to analyze the effects of that on communication systems. Specifically, a generated random height perturbation $H(x, y)$ based on user-defined $h_{rms}$ and $L_c$ values is translated to the corresponding additional phase variation on the surface as: $\phi_{rough}(x, y; h_{rms}, L_c) = 2\pi \frac{H(x,y)}{\lambda}$. Therefore, the electric field reflected from the surface of a rough object can be approximated as:

$$E_{vir}(x_0, y) = E(x_{ref}, y_{ref}) \times e^{j2\pi\phi_{rough}(x,y;h_{rms},L_c)}$$

Hence, the reflection off of a rough surface can be characterized similar to a smooth reflector, albeit by updating the electric field at the virtual source according to the equation above.

### Implementation of Reconfigurable Surfaces in NirvaWave

Reconfigurable Intelligent Surfaces (RIS) are emerging as a promising technology for future wireless communication systems. RIS employs a large number of low-cost passive unit cells to intelligently and flexibly control electromagnetic wave properties—such as amplitude, phase, and polarization—thereby optimizing the propagation channel to establish the best possible transmission links. Therefore, implementing near-field channel modeling to analyze THz and sub-THz wireless systems in the presence of RIS would be highly important. To implement RIS, we need to change the phase and amplitude configuration of the calculated E-field on the RIS plane, $E(x_{RIS}, y_{RIS})$, based on the user-defined RIS phase shift and amplitude for each element, denoted by $\phi_{RIS}$ and $A_{RIS}$. Specifically, we can write:

$$E_{vir}(x_0, y) = E(x_{RIS}, y_{RIS}) \times A_{RIS} e^{j2\pi\phi_{RIS}}$$

Using this approach, NirvaWave is able to model EM wave reflection off of the RIS by modifying the reflecting E-field distribution from $E_{vir}(x_0, y)$ to $E(x_{RIS}, y_{RIS})$.

### NirvaWave Recursive Algorithm

The environment may include several blockers, reflectors, or RISs. Hence, NirvaWave finds the wave propagation through a recursive algorithm, where the function is called within itself to account for the reflections off of the consecutive reflectors/RIS planes. This recursive algorithm first solves the EM wave propagation based on the initial electric field on the TX array and finds incident E-field on both sides of all reflector/RIS planes. This is done using a list of objects in the field of view and their geometric features. Then, for each reflector/RIS in the environment, the object list is updated using coordinate transformation and the contribution of the reflectors (as virtual sources) is calculated by repeating the same procedure. The algorithm would continue calculating the consecutive reflections until the termination condition is satisfied. Ultimately, the transformed E-fields resulting from the reflections and the TX radiation are recursively summed to obtain the total electric field in the environment.