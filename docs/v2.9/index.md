# V2.9.x — 2-Genome Neuroevolution (active)

## Overview

Neuroevolution with 4 parallel learning mechanisms. 2 independent genomes, 1 shared fitness. No reward function. Modular (8 modules), non-coding DNA, gene duplication. Open-ended through genome extension (each new capability = genome phụ ~10 dòng).

Core philosophy: Zero human intervention. Feature mới qua genome extension, ko qua module cứng. Fitness chỉ steps_alive + AE_loss_norm.

---

## V2.9.1 — ✅ Đang chạy

### Architecture
```
Agent:
  ├── Genome chính (100×8 params) → CPPN → policy + prediction weights
  └── 2nd genome (5 floats)        → dopamine: base[0:2] + sensitivity[3] + lr[4]
```

### 4 mechanisms (500 steps)
| Mechanism | Function | Gate |
|---|---|---|
| **GA** | Mutate + crossover genome | w_ga |
| **Gradient** | pred_error backprop → world model | w_grad × lr_grad |
| **Hebbian** | Synaptic plasticity | w_hebb |
| **Dopamine** | `softmax(base + [adapt,-adapt,0]), adapt=tanh(sens×error)` | — |

### Files
| File | Lines | Role |
|---|---|---|
| ae.py | 24 | Autoencoder 10→16→10, NaN-safe |
| cppn.py | 62 | CPPN → policy (30→10→8) + prediction (10→29) + 8 modular |
| env_ant.py | 56 | NoRewardAnt, 3 rings, 6 foods, energy=20, torque_cost=0.05 |
| genome.py | 121 | JIT mutate (scan), vmap crossover, Tag, 2nd-genome(5), modular, non-coding, dup |
| hebbian.py | 13 | Hebbian update with scale, returns 4 keys |
| main.py | 285 | Run loop: GA + gradient + hebbian + dopamine(adaptive) + VIP init + HF checkpoint |

### Environment
| Param | Value | Effect |
|---|---|---|
| energy_init | 20 | Baseline survival ~50 steps |
| energy_cost | 0.4 | Base cost per step |
| torque_cost | 0.05 | Movement penalty → valley of death |
| food_energy | 50 | +50 per eaten unit |
| rings | 3 (bk=5,10,15) | 6 food total, NO respawn |

### Status
- Valley of death confirmed: max 37-47, mean 33, ae_loss→0.005
- Dopamine emergence: GA 0.64-0.77 khi gradient/Hebbian chết
- Ko bug — CPPN ko thể output action=0 cho obs thay đổi → fitness < lý thuyết 50
- Đang chạy 5000 gen × 1024 pop

---

## V2.9.2 — VIP Init (genomic bottleneck) ✅ Code done, cần test

### Mục tiêu
Phá valley of death. Teacher gradient + curiosity → compress qua CPPN bottleneck → genome VIP → init pop.

### Cơ chế
Train teacher policy (gradient + curiosity, cùng env Ant) → teacher weights (w_ih 30×10, w_ho 10×8, w_pred 10×29) → optimize genome (100×8) minimize `||CPPN(genome) - teacher_weights||²` → init pop 1024 từ genome VIP ±5% mutate noise → V2.9.1 chạy bình thường.

### Files mới
| File | Lines | Role |
|---|---|---|
| `train_teacher.py` | ~70 | gradient + curiosity teacher, cùng env, cùng architecture |
| `vip_compress.py` | ~65 | tối ưu genome để khớp teacher weights qua CPPN bottleneck |

### CLI
```bash
# Step 1: Train teacher + compress genome
py main.py 5000 1024 3072 teacher

# Step 2: Run GA with VIP init (same code, different init)
py main.py 5000 1024 3072 vip vip_genome.npz
```

### Kiểm tra thành công
Fitness từ gen 1 ≥ 80. Ko stuck valley.

---

## V2.9.3 — Spatial Memory (grid/place cells) 🟡 Cần plan

### Mục tiêu
Agent nhớ multiple food sources, grid cell encoding từ `(x,y)`, multi-slot memory.

### Genome phụ (~8 floats)
| Param | Range | Paper |
|---|---|---|
| `grid_scale` | 0.5-2.0 | Grid cell period (Nature Comms 2025) |
| `place_radius` | 1.0-3.0 | Place field size |
| `n_slots` | 3-12 | Number of food memory slots |
| `slot_dim` | 4-16 | Encoding per slot |
| `recall_temp` | 0.5-2.0 | Slot selection softmax temp |
| `sweep_B` | 3-8 | Theta sweep beam (Nature 2025) |
| `sweep_L` | 5-20 | Theta sweep horizon |
| `reframe_threshold` | 0.3-0.8 | When to switch slot |

### Implementation reference
NEAT-NC (2026) — arXiv 2604.15076 — dùng place cells + grid cells + border cells + head direction cho NEAT path planning. REMI (NeurIPS 2025) — grid cell manifold + place cell autoassociation cho planning.

---

## V2.9.4 — Planning (beam search) 🟡 Cần plan

### Mục tiêu
Multi-step planning qua world model `w_pred` (đã train từ V2.9.1 gradient). MTP ICLR 2026 style: vmap B×L rollouts, JIT compatible.

### Genome phụ (4 floats)
| Param | Range | Meaning |
|---|---|---|
| `B_beam` | 3-10 | Beam width (number of trajectories) |
| `L_horizon` | 5-20 | Horizon per trajectory |
| `noise_scale` | 0.1-0.5 | Action noise |
| `exploit_rate` | 0.5-0.9 | Exploit vs explore |

### Cơ chế
```
Mỗi step:
  sample B action sequences × L steps
  vmap rollout qua w_pred → cumulative prediction error per sequence
  chọn sequence error thấp nhất → action đầu
```

### Paper ref
**MTP (ICLR 2026)** — model tensor planning, JAX + MJX, structured CEM. Code: `github.com/anindex/mtp`. **PRISM** — action prior từ learned encoder, Product-of-Gaussians fusion. **WEAVER** — Best-of-N, 5-10× faster.

### Dùng world model có sẵn
Ko cần pretrain — `w_pred` đã học từ V2.9.1 gradient. Planning chỉ rollout dài hơn.

---

## V2.9.5 — Self-Diagnosis (level 2, meta-regulation) 🟡 Dễ, đã có design

### Mục tiêu
Agent monitor rolling prediction error → tự adjust mutation rate dựa trên controllability estimate. Cấp thấp, ko attribution, ko counterfactual.

### Genome phụ (3 floats)
| Param | Range | Paper |
|---|---|---|
| `monitor_window` | 50-200 | PNAS 2026 mPFC metacontrol |
| `anomaly_threshold` | 1.5-3.0 | Std từ historical mean |
| `reflect_interval` | 50-500 | Steps between checks |

### Cơ chế
```python
window = pred_error_history[-monitor_window:]
err_mean = jnp.mean(window)
err_std = jnp.std(history) + 1e-8
if err_mean > jnp.mean(history) + threshold * err_std:
    mr *= 1.2  # explore — "mất kiểm soát"
elif err_mean < jnp.mean(history) - threshold * err_std:
    mr *= 0.9  # exploit — "ổn định"
```

### Implementation
~15 dòng trong main.py. Ko cần file mới.

---

## V2.9.6 — Imitation (level 3, mirror genome) 🟡 Cần design thêm

### Mục tiêu
Agent học từ elite trong population, có chọn lọc (3 tầng lọc).

### 3 tầng lọc
| Layer | Mechanism | Paper |
|---|---|---|
| Tầng 1 | `selectivity` threshold: chỉ imitate khi elite_fitness > agent × k | GANE (GECCO 2023) |
| Tầng 2 | Alignment ở hidden state (ko copy action — copy ý định) | Contrastive IL (2025) |
| Tầng 3 | Self-diagnosis gate: chỉ imitate khi "tao stuck" (từ V2.9.5) | — |

### Genome phụ (4 floats)
| Param | Range | Meaning |
|---|---|---|
| `proj_dim` | 32-128 | Projection dim for contrastive |
| `align_temp` | 0.07-0.5 | InfoNCE temperature |
| `selectivity` | 1.0-2.0 | Elite > agent × k |
| `align_lr` | 0.001-0.01 | Alignment learning rate |

### Paper ref
**GANE (GECCO 2023)** — generative adversarial neuroevolution, JAX. **EvIL (2024)** — evolution strategies for imitation, JAX + Brax. **Contrastive IL (ICCV 2025)** — bidirectional InfoNCE alignment.

---

## Roadmap thống nhất

```
V2.9.1 (done, valley 33) → V2.9.2 VIP init (code done, test) 
         → V2.9.3 Spatial memory (grid/place cells)
         → V2.9.4 Planning (beam search qua w_pred)
         → V2.9.5 Self-diagnosis (rolling error → mutation rate)
         → V2.9.6 Imitation (3-layer filtered mirror)
```

Mỗi bước = genome extension 10-50 dòng. Fitness giữ nguyên: `steps_alive + AE_loss_norm`. Zero human intervention. No reward, no RL.
