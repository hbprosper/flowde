# flowde: PyTorch-based solver for the reverse-time diffusion equation
## Reverse-Time Ordinary Differential Equation (ODE)

Let $x(t) \in \mathbb{R}^d$ be a $d$-dimensional vector with probability density $p(x)$ defined on the domain $t \in [0, 1]$.
Feng Bao *et al.*[1] note that the reverse-time ODE
```math
    dx  = \left[f(t) x - \frac{1}{2} g^2(t) S(t, x)\right] dt, \quad S(t, x) = \nabla_{x} \log p(x)
```
maps $x_1 \equiv x(t=1) \sim p(x_1)$ to $x_0 \equiv x(t=0) \sim p(x_0)$ *deterministically*, where $p(x_1)$ is a diagonal $d$-dimensional standard normal density and $p(x_0)$ is a desired $d$-dimensional target density for which it is assumed that a point cloud exists. The probability density $p(x)$ smoothly interpolates between $p(x_1)$ and $p(x_0)$ and $S(t, x)$ is the **score funtion** associated with $p(x)$.  Using the definitions of the functions $f(t)$ and $g(t)$ in Ref.[1], and after some manipulation, these equations can be written as
```math
\begin{aligned}
    \frac{d x}{dt} 
     & = \lambda(t) \, x + \mu(t)
     \, q(t, x), \text{ where}\\
   \lambda(t) & = \frac{d\log\sigma(t)}{dt}, \\
   \mu(t) & = \alpha(t) \frac{d\log\alpha(t)/\sigma(t)}{dt}, \text{ and }\\
q(t, x) & = \int_{\mathbb{R}^d}  x_0 \, p(x_0 \mid x) \, dx_0 ,\\
& =  \int_{\mathbb{R}^d}  x_0 \, \frac{p(x \mid x_0) \, p(x_0)}{p(x)} \, dx_0,\\
& =  \int_{\mathbb{R}^d}  x_0 \, p(x \mid x_0) \, p(x_0) \, dx_0 \, / \, \int_{\mathbb{R}^d}  p(x \mid x_0) \, p(x_0) \, dx_0, 
\end{aligned}
```
with $p(x \mid x_0)$ given by
```math
\begin{aligned}
p(x | x_0) & = {\cal N}(x; \, \alpha(t) x_0, \, \sigma^2(t) \mathbf{I}), \\
 & = \prod_{i=1}^d {\cal N}(x_{t, i}; \alpha(t) x_{0,i}, \sigma^2(t)),\\
 & = \frac{1}{(\sigma \sqrt{2\pi})^d} \exp\left[ -\frac{1}{2}\sum_{i=1}^d \left( \frac{x_{t, i} - \alpha(t) \, x_{0, i}}{\sigma(t)} \right)^2 \right], 
\end{aligned}
```
where $x_{t, i} \equiv x_i(t)$ is the $i^\text{th}$ component of the vector $x(t)$. Defining the $d$-dimensional vector
```math
\begin{align}
    z(t) & = \frac{x(t) - \alpha(t) \, x(0)}{\sigma(t)} ,
\end{align}
```
we can write $p(x \mid x_0)$ as
```math
\begin{aligned}
p(x | x_0) 
 & \propto \exp \left(-\frac{1}{2}  z^2 \right). 
\end{aligned}
```
The function $\alpha(t)$ is chosen so that it goes to zero as $t \rightarrow 1$, while $\sigma(t)$ remains finite in that limit. Then, by construction, the vector $x_1$ will be distributed according to a diagonal $d$-dimensional ormal with variance $\sigma_1^2$ irrespective of the density $p(x_0)$. The $d$-dimensional normal is a **fixed point** of the SDE. Reference [1] suggests $\alpha = 1 - t$, $\sigma(t) = \sqrt{t}$. In this project, we choose
```math
\begin{aligned}
\alpha(t) & = 1 - t,\\
\sigma(t) & = \sigma_1 t + \sigma_0 (1 - t) \quad\text{with } \sigma_1 = 1.
\end{aligned}
```
## Code
The class `FlowDE` numerically solves the equation
```math
\begin{aligned}
    \frac{d x}{dt} 
     & = \lambda(t) \, x + \mu(t)
     \, q(t, x),
\end{aligned}
```
where, following Ref.[1], the integrals that define the vector field $q(t, x)$ are approximated by Monte Carlo integration using point clouds $\sim p(x_0)$, which for intractable densities would usually be done via Monte Carlo simulation.

## References
 1. Yanfang Lui, Minglei Yang, Zezhong Zhang, Feng Bao, Yanzhao Cao, and Guannan Zhang, Diffusion-Model-Assisted Supervised Learning of Generative Models for Density Estimation, arXiv:2310.14458v1, 22 Oct 2023
