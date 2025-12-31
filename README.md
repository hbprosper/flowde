# flowde: PyTorch-based solver for the reverse-time diffusion equation
## Reverse-Time Ordinary Differential Equation (ODE)

Inline: $I(X;Z)$

Block:
$$
I(X;Z) = H(X) - H(X \mid Z)
$$

Even more remarkable than the above is the existence of a mechanism to go from $x_1$ to $x_0$.
In [2] Feng Bao *et al.* note the striking mathematical fact that the reverse-time ODE
$$
    dx_t  = \left[f(t) x_t - \frac{1}{2} g^2(t) S(t, x_t)\right] dt, \quad S(t, x_t) \equiv \nabla_{x_t} \log p(x_t),
$$
where $p(x_t)$ is the probability density of $x_t$ and $S(x_t, t)$ is the **score funtion** associated with $p(x_t)$ can be used 
to map $x_t$ back to $x_0$ *deterministically*.  This is remarkable! Moreover, because these $d$ equations are ordinary differential equations we can rewrite the above as
$$
    \frac{d x_t}{dt} &= G(t, x_t), \, \, \text{ where} \\
    G(t, x_t) & = f(t) x_t - \frac{1}{2} g^2(t) S(t, x_t) .
$$
