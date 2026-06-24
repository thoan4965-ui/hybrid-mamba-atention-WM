# V2.9.1 — 2-Genome Architecture (GA + Gradient + Hebbian + Dopamine)

## Overview

Neuroevolution with 4 parallel learning mechanisms. 2 independent genomes, 1 shared fitness. No reward function. Modular (8 modules), non-coding DNA, gene duplication.

## Files

| File | Lines | Role |
|---|---|---|
| ae.py | 24 | Autoencoder 10→16→10, NaN-safe |
| cppn.py | 62 | CPPN → policy (30→10→8) + prediction (10→29) + 8 modular |
| env_ant.py | 56 | NoRewardAnt, 3 rings, 6 foods, energy=20 |
| genome.py | 121 | JIT mutate (scan), vmap crossover, Tag, 2nd-genome(5), modular, non-coding, dup |
| hebbian.py | 13 | Hebbian update with scale (w_hebb), returns 4 keys |
| main.py | 237 | Run loop: GA + gradient + hebbian + dopamine(adaptive) + HF checkpoint |

## 2-Genome Architecture

```
Agent:
  ├── Genome chính (100×8 params) → CPPN → policy + prediction weights
  └── 2nd genome (5 floats)        → dopamine: base[0:2] + sensitivity[3] + lr[4]

Fitness = steps_alive + 5 × AE_loss_norm (cho cả 2 genome)
```

Each genome has independent crossover, mutation, init. Same fitness drives both.

## 4 Mechanisms (500 steps)

### GA (genome mutation + crossover)
- mutate: subst(nodes/conns bias/weight) + ins(node+conn) + dele(conn) + expr(non-coding) + module_id + dup(copy node+conns)
- crossover_innov: innovation-aligned, module-aware (same module → random pick, disjoint → fitter parent)
- JIT: mutate via lax.scan, crossover via vmap

### Gradient (world model)
- `pred_loss = ||next_obs - pred_next_obs||²` → jax.grad wrt w_ih, w_pred
- Update: `w -= lr_grad × w_grad × clip(grad, -1, 1)`

### Hebbian (online adaptation)
- `dw = w_hebb × η × outer(pre, post - tanh(post))`
- Clip: ±0.001 per step, weights bounded [-2, 2]

### Dopamine (adaptive coordinator)
```python
pred_error = ||next_obs - pred_next||²
adapt = tanh(sensitivity × pred_error)  # per-step adaptation
w_grad, w_hebb, w_ga = softmax(base + [adapt, -adapt, 0])
```

## Environment

| Param | Value | Effect |
|---|---|---|
| energy_init | 20 | Baseline survival ~50 steps |
| energy_cost | 0.4 | Base cost per step |
| torque_cost | 0.05 | Movement penalty (valley of death) |
| food_energy | 50 | +50 per eaten unit |
| rings | 3 (bk=5,10,15) | 6 food total, NO respawn |

## NaN Prevention

| Fix | Location |
|---|---|
| `jax_default_matmul_precision = 'high'` | main.py:3 |
| `mj_model.opt.iterations = 3` | main.py:16 |
| `nan_to_num(s2.obs)` | main.py:35 |
| `for k in pol: nan_to_num(pol[k])` | main.py:50 |
| `nan_to_num(d, 1.) / ex / dopa / nt` | main.py:56-155 |

## Checkpoint + Resume

- Save: every 500 gen → `checkpoints/v2.9/cp_{gen}.npz`
- Upload: HF `hhian/checkpoints/checkpoints/v2.9/cp_{gen}.npz`
- Resume: `download_latest_hf` → filter `checkpoints/v2.9/cp_*.npz` → load latest
- Backward compatibility: pad nodes 7→8, dope 3→5

## Open Architecture

New genomes can be added independently (sensor V2.7, body V2.8):
- 3 lines for init + 3 for crossover + 3 for mutate + 1 for checkpoint ≈ 10 lines/genome
- Independent evolution, shared fitness

## Limitation

- valley of death: torque_cost=0.05 makes movement cost more than standing. Fitness stuck ~40.

## Roadmap: V2.9.x (Neuroevolution + Learning)

| Version | Thêm | Paper ref | Thời gian |
|---|---|---|---|
| **V2.9.1** (hiện tại) | Modular(8), non-coding, dup, dopamine adaptive, 2-genome | — | ✅ Done |
| **V2.9.2** | **Spatial memory (grid cells)** — hippocampal-inspired place+grid cells for navigation | NEAT-NC 2025, REMI NeurIPS 2025 | ~2-3 ngày |
| **V2.9.3** | **Planning (working memory)** — recurrence + tree search B×L, state prediction + evaluator | MAP Nature Comms 2025, EvoPlan ICLR 2026 | ~5-7 ngày |
| **V2.9.4** | **Imitation learning** — mirror neurons + demo following, behavior cloning qua evolution | Mirror Neurons 2025, GANE 2024 | ~3-5 ngày |
| **V2.9.5** | **Sensory evolution (exaptation)** — gene duplication cho sensor, derived obs từ base | Organ Evolution Annual Reviews 2024 | ~3-5 ngày |

## V3 (separate project)

Multi-agent social: overhead camera, 2+ agents, social learning, cooperation/competition.

