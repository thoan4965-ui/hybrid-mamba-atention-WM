# Hybrid Mamba-2+Attention World Model

Hybrid block-level JEPA predictor cho robot manipulation. Thay MLP (LeWM AR) bằng Mamba-2 discrete state — vừa giữ temporal advantage, vừa không khuếch đại noise.

**Push-T: 94.7% ± 3.1% beat LeWM official 86.0% ± 4.0% (+8.7%) — cùng T4 fp32, 3 seeds.**

## Kết quả

| Task | Model | 3 seeds | Mean ± std |
|---|---|---|---|
| Push-T | **Hybrid Mamba-2** | 92, 98, 94 | **94.7% ± 3.1%** |
| Push-T | LeWM official | 82, 86, 90 | 86.0% ± 4.0% |
| TwoRoom | **Hybrid Mamba-2** | 84, 76, 96 | **85.3% ± 10.1%** |
| TwoRoom | LeWM official | 72, 78, 92 | 80.7% ± 10.3% |

## Thư mục

| Thư mục | Nội dung |
|---|---|
| `le-wm-v2.1/` | Hybrid Mamba-2+Attention — code chính |
| `le-wm-vo/` | V0: real robot bionic hand + CfC vs AR comparison |
| `le-wm-v1/` | V1: Hybrid CfC+Attention (abandoned, tham khảo) |

## Checkpoints & Data

HuggingFace: [hhian/checkpoints](https://huggingface.co/hhian/checkpoints)

Checkpoint epoch 10 là phiên bản cuối và tốt nhất cho tất cả các mô hình (Push-T, TwoRoom Mamba, TwoRoom CfC).

## Tham khảo

- [1] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
- [2] Maes, L. et al. (2026). LeWorldModel: Stable End-to-End JEPA from Pixels. *arXiv* 2603.19312.
- [3] Hasani, R. et al. (2022). Closed-form Continuous-time Neural Networks. *Nature Machine Intelligence*.
- [4] Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *ICML* 2024.
- [5] Balestriero, R. & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv* 2511.08544.
- [6] Ma, C. & Najarian, K. (2025). Rethinking the long-range dependency in Mamba/SSM and transformer models. *arXiv* 2509.04226.
- [7] Huang, Y. (2026). VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models. *arXiv* 2601.14354.
- [8] Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv* 2312.00752.
- [9] Lieber, O. et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model.
- [10] Li, Y. et al. (2026). TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model. *AAAI* 2026.
- [11] Liu, X. et al. (2020). Neural SDE: Stabilizing Neural ODE Networks with Stochasticity. *NeurIPS* 2020.
- [12] Rob Knight (2022). DexHand V1.0: Open-Source Dexterous Humanoid Robot Hand. GitHub.

