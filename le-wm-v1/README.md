# LeWM-V1 → V2: Hybrid Stateful Temporal Models for JEPA World Models

A world model for robotic manipulation, replacing the **stateless MLP** in LeWM's AR predictor with a **stateful temporal model** (CfC/Mamba-3) combined with Self-Attention.

**Key insight — orthogonal to existing hybrids:** Existing Attention-Mamba hybrids (Jamba, NVIDIA Hybrid) replace *Attention* (inter-token mixing) while keeping the *MLP* (per-token compute). We replace the *MLP* — the compute component — with a stateful temporal model (ODE CfC, Mamba-3) while keeping Attention for spatial feature extraction. These are two orthogonal design axes, and ours is unexplored for JEPA world models.

## Key contributions

- **Orthogonal hybrid design** — replacing MLP (stateless per-token compute) with stateful temporal model, not replacing Attention (unlike Jamba, NVIDIA Hybrid).
- **Hybrid ODE CfC-Attention predictor (V1)** — 6 blocks, CfC:Attention ≈ 1:1 ratio. CfC handles temporal dynamics, Attention handles spatial features. CfC is an ODE-based stateful model, replacing the stateless MLP in LeWM's AR predictor.
- **Scheduled sampling on ODE CfC** — reduces rollout error accumulation by training CfC to self-correct during autoregressive generation.
- **Denoiser MLP** — filters SIGReg regularization noise before feeding into CfC hidden state.
- **Full pipeline** — from simulation (TwoRoom, Push-T, Cube, Reacher) to real robot (8-DOF bionic hand, DexHand-based).

## Architecture

```
Input: 4 frames (T=4, frameskip=5)
  → TinyViT encoder → projector (MLP)
  → Denoiser MLP (residual)
  → 6× {Self-Attention (AdaLN) → ODE CfC}
  → pred_proj (MLP)
  → predicted embedding
```

**T=4 fixed for all benchmarks** (history_size=3, num_preds=1) — matching LeWM paper for fair architecture comparison.

| Component | Params |
|---|---|
| Attention | 787K/block (16 heads, dim_head=64) |
| ODE CfC | 764K/block (backbone_units=384) |
| AdaLN | 222K/block |
| **Total predictor** | **~10.6M** |

## Pre-trained models

### V1 hybrid (best)
| Benchmark | Model | Success rate |
|---|---|---|
| TwoRoom | Hybrid ODE CfC+Attn | **78%** |

### V0 bionic hand (best)
| Model | Dataset | Epoch | Loss |
|---|---|---|---|
| **ODE CfC V4** | `bionic_hand_dataset_v3_96.h5` (8900 frames, 89 episodes) | **30** | **0.002534** — best CfC |
| ODE CfC V3 | same | 40 | ~0.007 |
| AR | same | 30 | ~0.0012 |
| CfC rollout drift | — | — | **0.000014/step** (34× better than AR) |

*Full checkpoints uploaded to HuggingFace: [`hhian/checkpoints`](https://huggingface.co/hhian/checkpoints)*

## Dataset: bionic hand

The V0 model was trained on `bionic_hand_dataset_v3_96.h5` (143MB, 8900 frames, 89 episodes, 3 fingers × 8-DOF). The augmented version `bionic_hand_dataset_v3_96_aug.h5` (346MB, 17800 frames) was created later via 50-50 mix with augmented frames (ColorJitter per-sequence). Both datasets are available on HuggingFace.

| Benchmark | Architecture | Success rate | Status |
|---|---|---|---|---|
| TwoRoom | AR (LeWM paper) | 87% | ✅ Published |
| TwoRoom | **Hybrid ODE CfC+Attn (V1)** | **78%** | ✅ Done — CfC sensitive to SIGReg noise |
| TwoRoom (V2 hypothesis) | **Mamba-3+Attention** | **>87% (target)** | 🔄 Developing — theory supports, see below |
| TwoRoom (λ sweep) | Hybrid ODE CfC+Attn | TBD | 🔄 Planned (RTX 5080 BF16) |
| Push-T | TBD | TBD | 📅 |
| Cube | TBD | TBD | 📅 |
| Reacher | TBD | TBD | 📅 |

## V2: Mamba-3+Attention (in development)

**Hypothesis (theory-driven):** Replacing CFc (ODE-based) with Mamba-3 (selective SSM) will:
1. Eliminate SIGReg noise accumulation through ODE hidden states (Mamba uses discrete states)
2. Provide better temporal memory via input-dependent selectivity (Δ controls forgetting rate)
3. Beat LeWM's AR predictor (87%) at T=4 with the same training config

**Theoretical support:**
- Mamba-3 (Lahoti et al. 2026, ICLR 2026 Oral): complex-valued SSM + exponential-trapezoidal discretization + MIMO — state-of-the-art sub-quadratic LM
- Understanding Input Selectivity in Mamba (2506.11891): S6 memory decay can be dynamically slowed via "freezing time" — input-dependent Δ mitigates exponential forgetting
- NVIDIA Mamba-2-Hybrid (Waleffe et al. 2024): Hybrid SSM-Attention beats Transformer on ALL 12 benchmarks (+2.65 avg)
- Drama (Wang et al. 2024, ICLR 2025): Mamba-2 world model (Dreamer-style), 7M params, trains on laptop
- No existing paper: replaces MLP with Mamba in JEPA predictor, or compares CfC vs Mamba head-to-head for world models

**Status:** Code in development. Plan: debug on Colab T4 → RTX 5080 training → 4 benchmark evaluation (TwoRoom, Push-T, Cube, Reacher).

**Design:**
```
V2:  T=4 frames → TinyViT encoder → projector → 6×{Self-Attn(AdaLN) → Mamba-3} → pred_proj
     ↑ Keeps Attention for spatial mixing    ↑ Replaces MLP with stateful SSM
```

## Data

Download datasets from [HuggingFace](https://huggingface.co/collections/quentinll/lewm):

```bash
tar --zstd -xvf tworoom.tar.zst -C /path/to/data
export STABLEWM_HOME=/path/to/data
```

## Training

```bash
python train.py --config-name=lewm_hybrid data=tworoom loss.sigreg.weight=0.01 subdir=lambda_0_01
```

Override for BF16 (RTX 5080, L40S):
```bash
python train.py --config-name=lewm_hybrid ++trainer.precision=bf16-mixed ++loader.num_workers=6 data=tworoom loss.sigreg.weight=0.01 subdir=lambda_0_01
```

## Logbook

Detailed development log at `LOGBOOK.md` (~2900 lines) — daily records of decisions, bugs, and lessons learned.

## Hardware

- **Training:** NVIDIA L40S, RTX 5080 (BF16 native), Colab T4 / Kaggle GPU (FP16)
- **Real robot:** 8-DOF bionic hand (DexHand V1-based), SC09 bus servos, RP2350 controller

## Acknowledgments

- [LeWM](https://github.com/lucas-maes/le-wm) — original world model paper and codebase (Maes, Le Lidec, Scieur, LeCun, Balestriero)
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining) — training framework
- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) — environment and planning framework
- [DexHand V1](https://github.com/microsoft/DexHand) — open-source bionic hand design
- [ncps](https://github.com/mlech26l/ncps) — CfC / ODE-RNN implementation

## Citation

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
