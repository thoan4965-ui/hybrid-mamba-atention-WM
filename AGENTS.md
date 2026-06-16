# LeWM Project — AGENT CONTEXT (CURRENT: V2.1 Mamba-2+Attention)

## ⚡ MASTER PROTOCOL — 3 pillars of truth + 3-step decision

### 3 pillars of truth (cho mọi kết luận)
Mọi kết luận phải đủ: **Lý thuyết (Theory) + Paper/Data + Thực nghiệm (Empirical)**. Thiếu 1 → ghi "chưa biết, cần X".

### 3-step decision (cho mọi hành động)
1. **Test trước** — kiểm tra data có đủ ko? Có mơ hồ ko? Nếu thiếu data → dừng, báo "thiếu data", ko đoán.
2. **Chọn đường tối ưu bền vững** — ko chỉ "chạy được là được". Chọn giải pháp robust, maintain được lâu dài, ko technical debt.
3. **Mơ hồ = thiếu data** — ko thể kết luận chắc chắn = thiếu thông tin. Nói thẳng "chưa đủ data để kết luận, cần X". Ko suy diễn, ko bịa.

### Áp dụng
- Trước mỗi lệnh cài đặt: check CUDA+Torch, check wheel có sẵn ko, dataset có public ko
- Trước mỗi train: verify config, data path, checkpoint resume
- Trước mỗi kết luận: đủ 3 pillars chưa? Nếu chưa → báo thiếu

## Project Scale
- **V0 [done]:** 1 bionic hand 8-DOF real, grasp confirmed
- **V1 [abandoned]:** Hybrid CfC+Attention TwoRoom — 78% (T=16 sai lầm, đáng lẽ T=4)
- **V2.1 [active]:** Mamba-2+Attention predictor, TwoRoom (đang train trên Vast)
- **Social T1 [future]:** 1 overhead cam, joint latent, 2×SO-ARM100 (MuJoCo)
- **Social T2 [future]:** 3 cam (overhead + 2 ego), cross-attn

## V2.1 Architecture
- **6×{Self-Attn(AdaLN) → Mamba-2}**, T=4, heads=6, d_state=64
- Attention:Mamba-2 ≈ 1.3:1 params (288K:221K)
- Mamba-2 (có wheel sẵn, ko build), Mamba-3 giữ trong module.py để tham khảo

## Strategy: Fair architecture comparison (fixed T=4)
- LeWM paper dùng T=4 cho cả 4 benchmark
- Mọi architecture dùng **cùng T=4, batch=128, seed=3072**
- Ai beat LeWM ở T=4 → architectural improvement thật

## Priorities
1. ✅ V2.1 Mamba-2 code + config + all-in-one script
2. 🔄 V2.1 train TwoRoom trên Vast (RTX 5080, $0.175/h)
3. 📅 V2.1 eval TwoRoom + Push-T + Cube + Reacher
4. 📅 Báo cáo Sáng tạo trẻ 30/6 (V0 + V2.1)
5. 📅 ISEF tháng 9 (cần budget ~$72-120)

## Rules từ bug
1. **Resume:** `glob *.ckpt`, ko hardcode filename
2. **HF upload:** `{subdir}/{run_name}/ep_{epoch}` — ko hardcode
3. **Check CUDA+Torch trước cài wheel:** `python -c "import torch; print(f'{torch.__version__}, CUDA: {torch.version.cuda}')"`
4. **Ko build source nếu có wheel:** causal-conv1d, mamba-ssm đều có wheel
5. **Dataset .tar.zst → download + giải nén thủ công** (download_data.py)
6. **Eval seed:** seed=3072 đồng bộ train
7. **Precision:** bf16 trên GPU hỗ trợ
8. **CEM context history bug:** phải truyền context_emb, ko dùng 1 frame
9. **Ko tự bịa threshold/cơ chế** — phải có test thật

## Budget
- Vast RTX 5080: $0.175/h (đang dùng)
- Colab T4: free (nhiều acc)
- Kaggle GPU: free (nhiều acc)

## Work Protocol
- Output mỗi tool call
- Ko sub-agent code
- Manager → worker
- Mày nói "dừng" → tao stop ngay
