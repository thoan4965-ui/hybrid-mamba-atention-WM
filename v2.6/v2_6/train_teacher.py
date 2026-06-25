"""Teacher: gradient + curiosity. Train policy in same env, same architecture."""
import jax, jax.numpy as jnp, time
jax.config.update('jax_default_matmul_precision', 'high')
from jax import random, jit, vmap, grad
from v2_6.cppn import policy_forward
from v2_6.env_ant import NoRewardAnt

# Same env as V2.9.1
env = NoRewardAnt(backend='mjx', energy_init=20., energy_cost=0.4, torque_cost=0.05)
if hasattr(env, 'sys') and hasattr(env.sys, 'mj_model'):
    env.sys.mj_model.opt.iterations = 3
    env.sys.mj_model.opt.ls_iterations = 5
    env.sys.mj_model.opt.timestep = 0.001

def init_teacher(key):
    """Init teacher policy weights (same dims as CPPN output)."""
    k1, k2, k3 = random.split(key, 3)
    return {
        'w_ih': random.normal(k1, (30, 10)) * 0.1,
        'w_ho': random.normal(k2, (10, 8)) * 0.1,
        'w_pred': random.normal(k3, (10, 29)) * 0.1,
    }

@jit
def teacher_loss(params, obs_seq, action_seq, next_obs_seq):
    """Loss = pred_error - β × action_entropy (curiosity)."""
    # Predict each step
    h = jnp.tanh(obs_seq @ params['w_ih'][:-1] + params['w_ih'][-1])
    pred_next = h @ params['w_pred']
    pred_err = jnp.mean((next_obs_seq - pred_next) ** 2)

    # Curiosity: estimate action distribution entropy
    h_ho = jnp.tanh(h @ params['w_ho'])
    action_dist = jnp.tanh(h_ho)
    action_var = jnp.var(action_dist) + jnp.var(action_seq)
    curiosity = -0.01 * action_var

    return pred_err + curiosity

def rollout(key, params, n_steps=200):
    """Run 1 episode, collect trajectory. Short ~200 steps for quick teacher."""
    s = env.reset(key)
    obs_list, act_list, nxt_list = [], [], []
    for _ in range(n_steps):
        a, pred_n = policy_forward(params, s.obs)
        s2 = env.step(s, a)
        s2 = s2.replace(obs=jnp.nan_to_num(s2.obs, 0.))
        obs_list.append(s.obs)
        act_list.append(a)
        nxt_list.append(s2.obs)
        if s2.done > 0.5: break
        s = s2
    return jnp.stack(obs_list), jnp.stack(act_list), jnp.stack(nxt_list)

def train_teacher(n_episodes=100, lr=0.001, seed=3072):
    """Train teacher policy with gradient + curiosity."""
    key = random.PRNGKey(seed)
    params = init_teacher(key)

    for ep in range(n_episodes):
        k_ep = random.fold_in(key, ep)
        obs, act, nxt = rollout(k_ep, params)
        loss, grads = jax.value_and_grad(
            lambda p: teacher_loss(p, obs, act, nxt))(params)
        params = {k: params[k] - lr * jnp.clip(grads[k], -0.5, 0.5)
                  for k in params}
        if (ep + 1) % 20 == 0:
            # Eval: run a clean rollout
            _, _, _ = rollout(random.fold_in(key, ep + 9999), params)
            alive = 0
            for _ in range(5):
                obs_e, _, _ = rollout(random.fold_in(key, ep + alive), params)
                alive += obs_e.shape[0]
            alive /= 5
            print(f"  Teacher ep{ep+1}: loss={float(loss):.4f} avg_steps={alive:.0f}")
    return params
