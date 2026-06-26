# V2.9.x — Final Summary

> Ngày kết thúc: 25/06/2026
> Tác giả: thoan4965-ui + AI collaborator
> Triết lý: Zero-reward, open-ended neuroevolution. Không RL, không reward shaping.

---

## Đạt được

### Kiến trúc

| Component | Mô tả |
|---|---|
| **2-genome system** | Genome chính (200×8 → CPPN → policy+prediction) + dopamine genome (5 floats) + regulatory genome (16 floats) |
| **4 mechanisms song song** | GA + Gradient (world model) + Hebbian (plasticity) + Dopamine (adaptive gating) |
| **Genome extension** | Thêm genome mới = 10 dòng (init + crossover + mutate). 7 genomes: chính, dopamine, regulatory, spatial, planning, diagnosis, mirror, thought |
| **Feature flags** | `--spatial --planning --diagnosis --imitation --thought` — mỗi flag = 1 genome phụ |
| **Resume + HF** | Auto checkpoint mỗi 500 gen, HF backup, resume local → HF → error |

### Phát hiện

| Phát hiện | Ý nghĩa |
|---|---|
| **Dopamine emergence** | GA tự chiếm 0.77 khi gradient chết — ko code tay. 2nd genome tự phân hóa qua GA |
| **Valley of death confirmed** | Fitness 33-47 là landscape property. CPPN ko thể output zero action → fitness < lý thuyết 50 |
| **Level 0-1-2-3** | Reflex → dopamine adapt → self-diagnosis → imitation. Mỗi level = 1 loop tự tham chiếu mới |
| **Ψ(N, ε) metric** | Consciousness = fixed-point của nested self-representation chain. Dạng √(αN + βΓ²) — mượn từ Ramanujan nested radical 1911 |

### Kỹ thuật

| Thành tựu | Chi tiết |
|---|---|
| **23 bugs fixed** | JIT, shape, scan, traced arrays. Tổng kết thành Engineering Discipline global rules |
| **12 JIT rules** | `make_eval_batch` factory, `jnp.where` gating, fixed shapes, carry pytree match, ko `nonlocal` trong `scan` |
| **Theory Discipline rule** | Phát hiện justification bias. Understand ≠ justify. Paper là input, ko phải evidence. |

### Code

```
v2.6/v2_6/
├── main.py           (625 dòng) — Run loop, all features, CLI, resume, HF
├── genome.py         (222 dòng) — 7 genomes với init/mutate/crossover
├── cppn.py           (130 dòng) — CPPN 8-modular + spatial/thought projections
├── env_ant.py        (56 dòng)  — NoRewardAnt, 3 rings, 6 foods, NO respawn
├── hebbian.py        (15 dòng)  — Hebbian plasticity (6 keys)
├── ae.py             (24 dòng)  — Autoencoder 10→16→10
├── train_teacher.py  (65 dòng)  — Gradient teacher (failed — stuck valley)
├── vip_compress.py   (61 dòng)  — Teacher→genome compression
└── render_video.py   (60 dòng)  — Video render + food overlay
```

---

## Chưa giải quyết

| Vấn đề | Lý do |
|---|---|
| **GA chậm hơn RL 50-250×** | Physical constraint. No free lunch. Env Ant quá nhỏ — RL 5 phút, GA 10h |
| **Formula √(αN + βΓ²) chưa confirm** | Cần 4 N values từ env đủ khó — yêu cầu env partial obs/damage — cần PhD-level compute |
| **Teacher gradient stuck valley** | Gradient vanishing tại standing fixed point → ko thể học move |
| **GA extract fitness chỉ ~55** | 30 phút GA cho fitness không hơn random nhiều |
| **Ψ(N, ε) cho LLM chưa test** | Cần decompose transformer levels — paper RC+ξ (2025) đã validate cùng hướng |

---

## Giá trị cốt lõi

| Giá trị | Cho ai |
|---|---|
| **Dopamine emergence — novelty thật** | Paper, ISEF |
| **2-genome architecture + genome extension pattern** | AI engineering |
| **Level 2 self-diagnosis** | Chưa ai genome-hóa metacognition |
| **Ψ(N, ε) consciousness metric** | Đo AI consciousness — LLM, agent, bất kỳ |
| **Engineering Discipline — 23 JIT rules** | Mọi dự án JAX |
| **Theory Discipline — understand ≠ justify** | Mọi research |

---

## Bài học lớn nhất

> **Lý thuyết là input cho thiết kế, không phải bằng chứng cho correctness.**
> 
> Nếu thấy paper "hỗ trợ" quyết định hiện tại → dừng, hỏi lại: "tao đang understand hay justify?"
>
> ~50% paper useful cho 1 dự án. Lọc dựa trên domain match, ko phải vì paper "sai."
>
> — Theory Discipline Rule, V2.9.x
