# Papers Read — Organized by Version

## V0 — Bionic Hand 8-DOF (real robot, grasp demo)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-03-01 | LeWM: LeWorldModel | https://arxiv.org/abs/2603.19312 | JEPA end-to-end, TinyViT+AR, Push-T 96%, TwoRoom 87% | V0, V1, V2.1 |
| 2026-06-01 | RT2: Google DeepMind 2023 | https://robotics-transformer2.github.io/ | Vision-language-action model | V0 baseline |
| 2026-06-01 | DexHand Sim Resources | https://github.com/TheRobotStudio/V1.0-Dexhand | Open-source dexterous hand | V0 hardware |
| 2026-06-01 | SC09 Servo Datasheet | — | SC09 servo specs + calibration | V0 hardware |
| 2026-06-01 | IMU Proprioception Research | — | IMU-based hand pose estimation | V0 sensor |

## V1 — Hybrid CfC+Attention TwoRoom (abandoned, 78%)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-03-05 | CfC (Nature MI 2022) | https://doi.org/10.1038/s42256-022-00556-7 | ODE-RNN continuous hidden state | V1 core |
| 2026-03-10 | Drone Racing CfC (Sci Robotics 2023) | — | CfC OOD generalization for drone control | V1 ref |
| 2026-04-01 | Liquid-S4 (Hasani 2022) | https://arxiv.org/abs/2209.12951 | CfC gating + S4 state-space | V1 ref |
| 2026-05-01 | UDE (Rackauckas 2020) | https://arxiv.org/abs/2001.04385 | Universal Differential Equations | V1 theory |
| 2026-06-15 | Hybrid Transformer+LNN (Sci Reports 2025) | https://www.nature.com/articles/s41598-025-04210-1 | CfC+Attention hybrid works | V1 ref |

## V2.1 — Hybrid Mamba-2+Attention (main result, Push-T 94.7%)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-03-10 | Mamba (Gu & Dao 2023) | https://arxiv.org/abs/2312.00752 | Selective SSM, linear complexity | V2.1 |
| 2026-04-01 | Mamba-2 (Dao & Gu 2024, ICML) | https://arxiv.org/abs/2405.21060 | SSD layer, 2-8x faster, d_state=256 | V2.1 core |
| 2026-04-05 | Mamba-3 (ICLR 2026 Oral) | https://arxiv.org/abs/2603.15569 | Exp-trapezoidal, complex-valued, MIMO | V2.1 ref |
| 2026-04-10 | Jamba (AI21 2024) | https://arxiv.org/abs/2403.19887 | First hybrid SSM+Attention 1:7 | V2.1 ref |
| 2026-04-15 | NVIDIA Mamba-2-Hybrid (ICML 2024) | https://arxiv.org/abs/2406.07887 | 8B hybrid beat Transformer all benchmarks | V2.1 ref |
| 2026-04-20 | Hymba (ICLR 2025 Spotlight) | https://arxiv.org/abs/2411.13676 | Head-wise hybrid 1:1 ratio | V2.1 ref |
| 2026-05-01 | Drama (ICLR 2025) | https://arxiv.org/abs/2410.08893 | Mamba-2 world model Atari, 7M params | V2.1 related |
| 2026-06-20 | TransMamba (AAAI 2026) | https://arxiv.org/abs/2503.24067 | Sequence-level dynamic hybrid switching | V2.1 ref |

## V2.5 — 4-DOF Robot Demo (lightweight deploy, edge MCU)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-05-10 | CompACT (2025) | https://arxiv.org/abs/2503.03062 | Compress visual obs→8 tokens, 40x speedup | V2.5 core |
| 2026-06-21 | MambaVision (CVPR 2025) | https://arxiv.org/abs/2407.08083 | Hybrid Mamba-Transformer vision backbone | V2.5 encoder |
| 2026-06-21 | Quamba (ICLR 2025) | https://arxiv.org/abs/2501.16796 | Post-training quantization for Mamba, ONNX INT8 | V2.5 deploy |
| 2026-06-21 | SLiM (DeepMind ICML 2025) | — | Quantization + pruning + low-rank combined | V2.5 ref |
| — | Vim / VMamba (ICML 2024) | https://arxiv.org/abs/2401.09417 | Pure SSM vision backbone (bidirectional Mamba) | V2.5 encoder |
| — | TinyViT (2022) | https://arxiv.org/abs/2211.07931 | Efficient vision transformer 21M→6M | V2.5 encoder |
| — | Knowledge Distillation (Hinton 2015) | https://arxiv.org/abs/1503.02531 | Distill large model into small | V2.5 deploy |

## V2.6 — Neuroevolution + Genomic Bottleneck (Action Only ✅ active)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| — | NEAT (Stanley 2002) | https://doi.org/10.1162/106365602320169811 | Evolving NN topologies via GA | V2.6 core |
| — | HyperNEAT (Stanley 2009) | https://doi.org/10.1145/1569901.1569903 | CPPN indirect encoding for large NN | V2.6 core |
| — | CPPN (Stanley 2007) | https://doi.org/10.1016/j.neunet.2007.09.016 | Compositional Pattern Producing Networks | V2.6 core |
| — | Genomic Bottleneck (Zador 2019) | https://doi.org/10.1038/s41583-019-0212-5 | Genome ~1:1M compression; DNA encodes rules not synapses | V2.6 theory |
| 2024 | Genomic Bottleneck Innate Ability (Zador PNAS) | https://www.pnas.org/doi/abs/10.1073/pnas.2409160121 | NN weights compress 1000× through bottleneck | V2.6 theory |
| 2025 | TensorNEAT (Wang 2025) | https://doi.org/10.1145/3730406 | JAX GPU NEAT/CPPN/HyperNEAT, 500× speedup | V2.6 infra |
| 2021 | Evolve & Merge (Najarro & Risi 2021) | https://dl.acm.org/doi/10.1145/3449639.3459317 | Hebbian rules + genomic bottleneck | V2.6 Hebbian |
| 2020 | Hebbian Meta-Learning (Najarro 2020, NeurIPS) | https://proceedings.neurips.cc/paper/2020/file/ee23e7ad9b473ad072d57aaa9b2a5222-Paper.pdf | Hebbian in random nets, no reward | V2.6 Hebbian |
| 2023 | Lamarckian Robot Evolution (Nature 2023) | https://www.nature.com/articles/s41598-023-48338-4 | Lamarckian inheritance in robot evolution | V2.6 tag |
| 2024 | Lamarckian Dynamic Env (arxiv 2024) | https://arxiv.org/abs/2403.19545 | Lamarckian > Darwinian in changing envs | V2.6 tag |
| 2023 | Epigenetic Opportunities (PMC 2023) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10170609/ | Epigenetic tags guide mutation, inheritable | V2.6 tag |
| — | Deep Neuroevolution (Such 2017) | https://arxiv.org/abs/1712.06567 | GA scales to DNN | V2.6 ref |
| — | Evolution Strategies (Salimans 2017) | https://arxiv.org/abs/1703.03864 | ES as RL alternative | V2.6 ref |
| — | Information Bottleneck (Tishby 2015) | https://arxiv.org/abs/1503.02406 | Bottleneck forces generalization | V2.6 theory |

## V2.9.x — 2-Genome Neuroevolution (active)

✅ = useful — có genome param cụ thể. ⚠️ = tham khảo design. ❌ = ko áp dụng.

| Date | Title | Link | Key findings | Verdict |
|---|---|---|---|---|
| 2025 | Grid Cells — Comp Cognitive Map (Nature Comms) | https://www.nature.com/articles/s41467-025-62733-7 | Object vector cells (30% MEC) = food detector. Grid cells = metric baseline. Compositional. | ✅ V2.9.3 — genome param: `n_slots`, `slot_dim`, `recall_temp` |
| 2025 | Grid Cells — Multiple Reference Frames (Nature Neuro) | https://www.nature.com/articles/s41593-025-02054-6 | Grid cells reanchor to task objects, NOT global GPS. | ✅ V2.9.3 — genome param: `reframe_threshold` |
| 2025 | Grid Cells — Trajectory Code (eLife) | https://elifesciences.org/articles/96627 | Grid cells encode 2D trajectories (paths), not positions. | ✅ V2.9.3 — genome param: `traj_len` |
| 2025 | Theta Sweeps Left-Right (Nature) | https://www.nature.com/articles/s41586-024-08527-1 | Entorhinal-hippocampal quét không gian mỗi theta cycle. Phủ cả nơi chưa đến. | ✅ V2.9.3 — genome param: `sweep_B`, `sweep_L`, `sweep_theta` |
| 2025 | Vector-HaSH Episodic Memory (Nature) | https://www.nature.com/articles/s41586-024-08392-y | Grid cells = scaffold for associative memory. Memory palaces. | ✅ V2.9.3 — genome param: `place_radius`, `place_lr`, `max_places` |
| 2025 | Hebbian HC→MEC Plasticity (PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12324395/ | Hebbian từ HC→MEC từ từ tạo spatial map từ salient features. | ✅ V2.9.3 — genome param: `hc_hebb_lr`, `anchor_salience` |
| 2025 | Place Without Grid Cells (eLife) | https://elifesciences.org/articles/99302 | Place cells possible without grid if border + path integration. | ⚠️ V2.9.3 — fallback design nếu grid quá nặng |
| 2025 | Mirror Neuron Alignment (ICCV) | https://arxiv.org/abs/2509.21136 | Contrastive alignment obs↔action in shared latent. Bidirectional InfoNCE. | ✅ V2.9.6 — genome param: `proj_dim`, `align_temp`, `align_lr` |
| 2023 | GANE (GECCO) | https://arxiv.org/abs/2304.12432 | Adversarial neuroevolution imitation. Co-evolve gen+disc. NE matches pre-trained. | ✅ V2.9.6 — genome param: `selectivity` (adversarial threshold) |
| 2026 | mPFC Metacontrol (PNAS) | https://www.pnas.org/doi/abs/10.1073/pnas.2510334123 | Prediction error → controllability estimate → adjust strategy. | ✅ V2.9.5 — genome param: `monitor_window`, `anomaly_threshold`, `reflect_interval` |
| 2026 | 3-Component Metacognition (Sci Rep) | https://link.springer.com/article/10.1038/s41598-026-37612-w | Monitor (dPFC) → Control (aPFC) → Decision (vFP). Confidence signals. | ⚠️ V2.9.5 — architecture tham khảo cho 3-stage self-diagnosis |
| 2025 | Intrinsic Metacognition (PMLR) | https://proceedings.mlr.press/v267/liu25cw.html | 3 components: knowledge, planning, evaluation. Current agents = extrinsic. | ⚠️ Triết học cho V2.9.x: intrinsic > extrinsic |
| 2024 | Fitness Valley (J Math Bio) | https://link.springer.com/article/10.1007/s00285-024-02143-3 | Crossing rate ∝ K·μ^L. Valley width L → time scale. | ✅ Design tool — tính L từ số gen stuck, determine mutation rate |
| 2025 | Pit Stops (arxiv) | https://arxiv.org/abs/2503.19766 | Intermediate positive-fitness mutants accelerate crossing. Pit stop. | ✅ V2.9.2 VIP init = pit stop — giảm L từ ≥3 xuống 0 |
| 2025 | Epistatic Hotspots (PNAS) | https://www.pnas.org/doi/10.1073/pnas.2413884122 | Sparse epistatic mutations boost evolvability. Suboptimal peaks = stepping stones. | ✅ V2.9.1 — non-coding DNA = epistatic hotspot, tăng dup rate |
| 2024 | Network Pop Valley (PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11151934/ | Pop structure determines valley crossing. Low accel+amplif = better. | ⚠️ Xác nhận flat tourn selection ≈ optimal, ko cần thay đổi |
| 2025 | Dominated Novelty Search (arxiv) | https://arxiv.org/abs/2502.00593 | QD as GA with dynamic fitness. No params, no grid. SOTA. | ⚠️ Reference nếu cần dynamic fitness, chưa cần ngay |

### Papers đã loại (ko useful)
| Title | Lý do |
|---|---|
| dACC = Meta-RL (PLOS Comp Bio 2025) | RL-based meta-learning, V2.9.1 zero reward |
| Meta-Dyna (Frontiers 2025) | Q-learning/Dyna-Q core, RL-based |
| Robotic MNS UBAL (ICANN 2024) | Camera + visual pipeline, iCub-specific grasping |
| ASAL (2024) | Foundation model inference quá nặng cho T4, task khác |

## V2.7 — + Sensor Evolution (upcoming, kế thừa V2.6)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2025 | Policy Manifold Search (Rakicevic) | https://github.com/nemanja-rakicevic/policy_manifold_search | AE for policy latent space | V2.7 AE |
| — | Weight-Agnostic NN (Gaier & Ha 2019) | https://arxiv.org/abs/1906.04358 | Random weights + good architecture | V2.7 ref |

## V2.8 — + Body Evolution (future, kế thừa V2.6+V2.7)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| — | POET (Wang 2019) | https://arxiv.org/abs/1901.01753 | Paired Open-Ended Trailblazer | V2.8 ref |
| — | Open-Ended Learning (OpenAI 2021) | https://arxiv.org/abs/2107.12808 | Open-ended learning | V2.8 vision |

## V3 — Multi-agent Future (overhead cam, 2 robots, social)

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-05-15 | COMBO (ICLR 2025) | https://proceedings.iclr.cc/ | Multi-agent world model | V3 core |
| 2026-05-20 | S3AP Social World Models | https://arxiv.org/abs/2509.00559 | Social reasoning via structured rep | V3 core |
| 2026-06-21 | V-JEPA (Meta 2024) | https://arxiv.org/abs/2404.08471 | Video JEPA, feature prediction, ViT-H/16 | V3 encoder |
| 2026-06-21 | V-JEPA 2 (Meta 2025) | https://arxiv.org/abs/2506.09985 | 1B param video JEPA, action-conditioned | V3 encoder |
| 2026-06-21 | SO-ARM100 / SO-101 | https://github.com/TheRobotStudio/SO-ARM100 | Open-source 6-DOF robot arm, $200-300 | V3 hardware |
| — | CLIP (Radford 2021, ICML) | https://arxiv.org/abs/2103.00020 | Contrastive joint embedding, 400M image-text pairs | V3 alignment |
| — | MDN (Bishop 1994) | https://publications.aston.ac.uk/id/eprint/373/ | Mixture Density Networks, multi-modal prediction | V3 predictor |
| — | VAE / KL (Kingma 2014, ICLR) | https://arxiv.org/abs/1312.6114 | Variational Bayes, KL divergence closed-form | V3 regularization |

## Baselines & Theory

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-06-01 | DINO-WM (Zhou et al. 2024) | https://arxiv.org/abs/2411.04983 | World model with DINOv2 encoder | Baseline |
| 2026-06-20 | PLDM (2024) | https://arxiv.org/abs/2311.00978 | End-to-end JEPA with VICReg 7-term loss | Baseline |
| 2026-06-20 | JEPA (LeCun 2022) | https://openreview.net/forum?id=BZ5a1r-kVsf | Joint Embedding Predictive Architecture | Theory |
| 2026-06-21 | Mamba Theory (4 papers) | See link-paper/32 | Selectivity, memory, hidden attention, limits | Theory |
| 2026-03-10 | Scheduled Sampling (Bengio 2015) | https://arxiv.org/abs/1506.03099 | Schedule curriculum for autoregressive models | General |
| 2026-06-01 | Symbolic Models (Cranmer 2020) | https://arxiv.org/abs/2006.11287 | Discover physics from data | General |

## Tools & Hardware

| Date | Title | Link | Key findings | Used in |
|---|---|---|---|---|
| 2026-06-10 | infomeasure (Nature 2025) | https://arxiv.org/abs/2505.14696 | Unified info theory toolkit, Nature 2025 | Tools |
| — | MuJoCo | https://mujoco.org/ | Physics engine for robot simulation | Tools |

## Competition

| Date | Title | Link | Key findings |
|---|---|---|---|
| 2026-06-21 | ISEF References | See link-paper/35 | Competition categories, rules, timeline |
