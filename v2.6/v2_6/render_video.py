"""Render video from saved .npz model — chạy 1 cell duy nhất"""
import os, time, subprocess
os.environ['MUJOCO_GL'] = 'glx'
os.environ['DISPLAY'] = ':99'

# --- Import ---
from v2_6.main import env, genome_to_policy, policy_forward
import jax, jax.numpy as jnp, numpy as np, mediapy as media
from jax import lax, jit
import mujoco
from mujoco import mjx

DATA_PATH = 'v26_result.npz'  # file model
N_FRAMES = 200                # số frame
W = 320; H = 240             # resolution

# --- Load ---
data = np.load(DATA_PATH)
pol = genome_to_policy(jnp.array(data['best_nodes']), jnp.array(data['best_conns']))

# --- Step 1: Rollout GPU (fast) ---
@jit
def rollout(pol, key):
    def step(s, _):
        s2 = env.step(s, policy_forward(pol, s.obs))
        return s2, s2.pipeline_state
    _, states = lax.scan(step, env.reset(key), jnp.arange(N_FRAMES))
    return states

print("Rollout GPU...", flush=True)
t0 = time.time()
states = rollout(pol, jax.random.PRNGKey(0))
print(f"  {N_FRAMES} frames in {time.time()-t0:.1f}s", flush=True)

# --- Step 2: Render CPU (cached) ---
renderer = mujoco.Renderer(env.sys.mj_model, width=W, height=H)
frames = []
print("Render CPU...", flush=True)
t0 = time.time()

for i in range(N_FRAMES):
    mj_data = mjx.get_data(env.sys.mj_model, states[i])
    renderer.update_scene(mj_data)
    frames.append(renderer.render().copy())
    if (i+1) % 50 == 0:
        dt = time.time() - t0
        print(f"  {i+1}/{N_FRAMES}  {dt:.0f}s", flush=True)

renderer.close()
video = jnp.stack(frames)
print(f"  Done: {time.time()-t0:.0f}s, shape={video.shape}", flush=True)

# --- Show ---
media.show_video(video, fps=30, height=400)
