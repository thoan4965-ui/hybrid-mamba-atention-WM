"""Hebbian — with scale factor (w_hebb from dopamine)."""
import jax.numpy as jnp

def hebbian_update(params, obs, scale=1.0):
    pre_h = jnp.concatenate([obs, jnp.ones(1)])
    w_ih, w_ho = params['w_ih'], params['w_ho']
    post_h = jnp.tanh(pre_h @ w_ih)
    dw_ih = scale * 0.0001 * jnp.outer(pre_h, post_h - jnp.tanh(post_h))
    w_ih = jnp.clip(w_ih + jnp.clip(dw_ih, -0.001, 0.001), -2., 2.)
    post_o = jnp.tanh(post_h @ w_ho)
    dw_ho = scale * 0.0001 * jnp.outer(post_h, post_o - jnp.tanh(post_o))
    w_ho = jnp.clip(w_ho + jnp.clip(dw_ho, -0.001, 0.001), -2., 2.)
    return {'w_ih': w_ih, 'w_ho': w_ho, 'w_pred': params['w_pred'],
            'w_dopa': params['w_dopa'],
            'w_spat': params['w_spat'], 'w_thought': params['w_thought']}
