"""Differentiable scalar physical-angle inversion for sheared exact surfaces."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from analytic import shear_chart


def angle_cross(t, k, theta, phi, eps, S, lam):
    point = shear_chart(k, theta, t, eps, S, lam, xp=jnp)
    return point[0]*jnp.sin(phi)-point[1]*jnp.cos(phi)


@jax.custom_jvp
def physical_angle_root(k, theta, phi, eps, S, lam):
    """Return the unique chart angle t for physical cylindrical angle phi.

    The primal follows the original fixed-bracket bisection. The JVP solves
    the scalar implicit equation; it does not differentiate branch decisions.
    Inputs should be admissible and phi should avoid exact quadrant boundaries
    during local derivative experiments.
    """
    local_phi = jnp.mod(phi, 2*jnp.pi)
    lo = jnp.floor(local_phi/(jnp.pi/2))*(jnp.pi/2)
    hi = lo+jnp.pi/2

    def bisect(_, bracket):
        left, right = bracket
        mid = (left+right)/2
        cross = angle_cross(mid, k, theta, local_phi, eps, S, lam)
        return jnp.where(cross > 0, mid, left), jnp.where(cross > 0, right, mid)

    lo, hi = jax.lax.fori_loop(0, 54, bisect, (lo, hi))
    return (lo+hi)/2


@physical_angle_root.defjvp
def _root_jvp(primals, tangents):
    root = physical_angle_root(*primals)
    def equation(t, parameters):
        return angle_cross(t, *parameters)
    denominator = jax.grad(equation, argnums=0)(root, primals)
    numerator = jax.jvp(lambda parameters: equation(root, parameters),
                        (primals,), (tangents,))[1]
    return root, -numerator/denominator
