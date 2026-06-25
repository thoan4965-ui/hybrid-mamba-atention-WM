# Theory Foundation — V2.9.x Neuroevolution

> Phân tích paper theo Theory Discipline Rule:
> 1. Cơ chế gốc là gì?
> 2. Useful gì cho zero-reward neuroevolution?
> 3. Bỏ cái gì (biological detail ko liên quan artificial system)?

---

## 1. Spatial Navigation

### 1.1 Grid Cells — Metric Space Encoding

**Cơ chế gốc (Nature Comms 2025):** MEC tạo compositional predictive map. Object vector cells (30% of MEC) fire gần object/landmark. Grid cells = baseline metric map. Kết hợp = biết object ở tọa độ nào.

**Useful V2.9.3:**
- Object vector slots → genome encoding multiple food positions. Agent hiện tại chỉ biết 1 food gần nhất (`dx, dy`). Ăn xong food đó → mù. Multi-slot cho phép nhớ pathway qua các ring food 5→10→15.
- Genome params: `n_slots` (số food slot, default=6 = số food), `slot_dim` (encoding dimension), `recall_temp` (softmax temperature khi chọn slot nào để follow)

**Ko useful:**
- Hexagonal firing pattern chi tiết — CPPN ko cần tái tạo hexagonal lattice
- 30% population ratio — chỉ cần N = số food object
- Phase precession, theta modulation — biological implementation detail

---

**Cơ chế gốc (Nature Neuro 2025):** Grid cells ko duy trì 1 global map. Chúng reanchor vào task-relevant objects, shift reference frame theo ngữ cảnh.

**Useful V2.9.3:**
- Khi ăn xong 1 food → reference frame shift sang food tiếp theo. Agent ko bị stuck ở 1 food source.
- Genome param: `reframe_threshold` — khi food cạn bao nhiêu % thì chuyển sang food kế

**Ko useful:**
- Orientation drift predicting homing direction — ko có homing behavior trong env
- Mismatch between grid phases — chi tiết nơ-ron

---

**Cơ chế gốc (eLife 2025):** Mathematical proof: grid cells encode 2-D trajectories, not positions. Cell sequences = path code.

**Useful V2.9.3:**
- Trajectory encoding thay vì position encoding. Agent ko cần biết "mình ở (x,y)" — cần biết "đi từ food1 → food2 mất bao nhiêu steps."
- Genome param: `traj_len` — trajectory encoding horizon (5-20 steps)

**Ko useful:**
- Chứng minh toán học chi tiết — interesting but ko ảnh hưởng design
- Cell sequence mechanism trong biological network

---

### 1.2 Theta Sweeps — Built-in Exploration

**Cơ chế gốc (Nature 2025):** Mỗi theta cycle (~8Hz), entorhinal-hippocampal quét không gian xung quanh, trái/phải luân phiên. Quét cả nơi chưa đến. Tồn tại cả trong REM sleep.

**Useful V2.9.3:**
- **Multi-step mental simulation — quan trọng nhất.** Agent hiện tại chỉ predict 1 step (prediction error). Theta sweeps = simulate B hướng, L steps mỗi hướng → chọn hướng tốt nhất. Có thể implement qua world model predictor `w_pred` đã train sẵn.
- Genome param: `sweep_B` (beam width = số hướng, 3-8), `sweep_L` (look-ahead steps, 5-20), `sweep_theta` (angular resolution, 4-8 hướng)

**Ko useful:**
- 8Hz frequency, theta cycle mechanism — implementation detail
- REM sleep persistence — agent ko ngủ
- Left-right alternation pattern — không cần thiết, uniform angular sampling đủ

---

### 1.3 Place Cells — Memory + Context

**Cơ chế gốc (Nature 2025 — Vector-HaSH):** Grid cells = low-dimensional scaffold. Place cells associate grid states với sensory experiences. Non-spatial episodic memory (memory palaces) xây trên spatial scaffold.

**Useful V2.9.3:**
- Grid cells làm scaffold cho place memory. Agent biết "ở vị trí A có food, ở vị trí B đã ăn rồi."
- Genome param: `place_radius` (kích thước place field, 1.0-3.0), `place_lr` (learning rate cho association, 0.01-0.1), `max_places` (số place cell tối đa, 10-50)

**Ko useful:**
- Memory palace mechanism — ứng dụng cho con người, ko cho agent ant
- Episodic memory detail (time cells, sequence replay) — too biological

---

**Cơ chế gốc (PMC 2025 — Slow Hebbian HC→MEC):** Hebbian plasticity từ hippocampus → grid cells từ từ tạo spatial map anchored vào salient features.

**Useful V2.9.3:**
- Hebbian update cho place→grid connection. V2.9.1 đã có Hebbian — extend để update spatial memory.
- Genome param: `hc_hebb_lr` (learning rate 0.0001-0.01), `anchor_salience` (salient feature threshold)

**Ko useful:**
- Slice electrophysiology evidence — biological method
- Temporally delayed plasticity mechanism

---

**Cơ chế gốc (eLife 2025 — Place without Grid):** Place cells có thể hình thành mà ko cần grid cells. Border cells + path integration đủ.

**Useful V2.9.x:**
- Nếu spatial memory genome quá nặng, có thể implement place-only memory (ko cần grid encoding) → nhẹ hơn, nhưng kém chính xác hơn.
- Design decision: implement grid encoding trước, nếu performance ko tốt → fallback place-only.

**Ko useful:**
- Debate về causal direction grid→place vs place→grid — ko ảnh hưởng engineering

---

## 2. Mirror Neurons — Imitation Learning

### 2.1 Embodied Representation Alignment

**Cơ chế gốc (ICCV 2025):** Independently trained action understanding model + execution model spontaneously develop aligned representations. Explicit contrastive alignment (bidirectional InfoNCE) + linear projections → shared latent space. Improves action recognition (+3.3%) and robot manipulation (+3.5%).

**Useful V2.9.6:**
- **Contrastive alignment** giữa agent representation và elite representation. Agent nhìn elite làm → so sánh với action nó định làm → alignment loss → học.
- **Khác biệt với paper:** Paper dùng 2 model riêng, pre-trained. V2.9.6 chỉ có 1 model (policy), elite là agent khác trong population.
- Genome param: `proj_dim` (projection dimension, 32-128), `align_temp` (InfoNCE temperature, 0.07-0.5), `align_lr` (learning rate 0.001-0.01)
- **Selectivity mechanism:** Chỉ align khi elite fitness > agent_fitness × selectivity. Tránh copy behavior tệ.

**Ko useful:**
- ViCLIP/ARP architecture — paper dùng model khác
- Video understanding backbone — ko liên quan neuroevolution
- Platonic Representation Hypothesis — interesting lý thuyết nhưng ko actionable

---

### 2.2 GANE — Neuroevolution Imitation

**Cơ chế gốc (GECCO 2023):** 2 populations co-evolve: generator (imitator) + discriminator (detector). Generator fitness = fool discriminator. Discriminator fitness = detect fake. Kết quả: generator matches pre-trained agent performance trên 8 Gym tasks.

**Useful V2.9.6:**
- **Adversarial imitation framework phù hợp với neuroevolution.** Khác 2-population, V2.9.6 chỉ cần 1 population + elite trajectory buffer. Generator = agent, discriminator = threshold function từ genome selectivity.
- Có thể coi mirror genome như 1 weak discriminator — phân biệt "giống elite hay ko" → điều chỉnh policy.

**Ko useful:**
- RNN architecture — V2.9.1 dùng CPPN
- Pre-trained RL agents làm target — V2.9.6 dùng elite trong population
- OpenAI Gym tasks — Ant env khác

---

### 2.3 Robotic MNS (UBAL)

**Cơ chế gốc (ICANN 2024):** Multi-layer connectionist model. Visual → motor mapping qua UBAL (Universal Bidirectional Associative Learning). Hebbian association between perception and action. For iCub/NICO humanoid.

**Ko useful — bỏ.**
- Yêu cầu camera + visual processing pipeline
- iCub/NICO robot-specific kinematics
- Grasping-specific action vocabulary
- V2.9.6 imitation khác: copy behavior từ elite trong pop, ko phải từ người qua camera

---

## 3. Metacognition — Self-Diagnosis

### 3.1 mPFC Predicts Action Success

**Cơ chế gốc (PNAS 2026):** Deep RL agents compute action prediction error in medial PFC. Error = "how well can I predict the outcome of my action?" If error consistently high → "controllability low" → need different strategy. Over/under-estimating controllability → depressive/anxious-like behaviors.

**Useful V2.9.5:**
- **Trực tiếp áp dụng được — quan trọng nhất của section này.** Self-diagnosis genome: rolling window prediction error → estimate controllability. Nếu controllability thấp (error cao liên tục) → increase mutation rate (explore). Nếu controllability cao (error thấp) → decrease mutation rate (exploit current strategy).
- Genome param: `monitor_window` (số steps để tính error trung bình, 50-200), `anomaly_threshold` (controllability threshold, 1.5-3.0 std), `reflect_interval` (how often to re-evaluate, 50-500 steps)
- **Novelty:** Chưa ai genome-hóa metacognition của mutation rate.

**Ko useful:**
- Over/under-estimation pathologies detail
- fMRI/neuroimaging biological evidence

---

### 3.2 3-Component Metacognition

**Cơ chế gốc (Sci Rep 2026):** 3 networks: monitoring (dPFC) → control (aPFC/FPC) → decision (vFP). Confidence signals in vmPFC/vACC. Representational uncertainty in dACC.

**Useful V2.9.5:**
- Architecture tham khảo cho 3-stage self-diagnosis: monitor (rolling error) → evaluate (compare threshold) → regulate (adjust mutation rate). Có thể implement như 3-step trong 1 genome.

**Ko useful:**
- Chi tiết từng brain region — V2.9.5 chỉ cần 1 genome module
- Resting-state connectivity analysis

---

### 3.3 Meta-Dyna

**Cơ chế gốc (Frontiers 2025):** Prefrontal meta-control + hippocampal mental simulation. Arbitration between model-free (habit) and model-based (planning). Dyna-Q framework extended with neural network world model.

**Useful V2.9.4 (Planning):**
- **Arbitration mechanism:** Khi nào dùng model-based planning (tree search) vs model-free (policy direct). V2.9.1 dopamine đã có arbitration giữa GA/gradient/Hebbian — extend cho planning vs no-planning.
- Mental simulation via rollouts = theta sweeps từ section 1.

**Ko useful:**
- Dyna-Q implementation detail — khác framework (tabula RL)
- Q-learning integration — V2.9.1 zero reward

---

### 3.4 Intrinsic Metacognitive Learning

**Cơ chế gốc (PMLR 2025):** Self-improving agents need 3 components: metacognitive knowledge (self-assessment), metacognitive planning (what/how to learn), metacognitive evaluation (reflect to improve). Current agents rely on extrinsic metacognition (human-designed loops).

**Useful V2.9.x philosophy:**
- **Đây là justification triết học cho "ko human intervention" rule.** V2.9.x càng về sau càng cần intrinsic metacognition (self-diagnosis genome tự quyết mutation rate). Tránh extrinsic (human tuning params mỗi khi stuck).
- Framework tham khảo cho V2.9.5 + V2.9.6: knowledge (self-diagnosis) → planning (spatial memory + theta sweep) → evaluation (imitation alignment loss).

**Ko useful:**
- LLM-based agents (tiêu điểm của paper) — khác hệ với neuroevolution
- Concrete implementation pattern — paper là framework lý thuyết

---

## 4. Fitness Landscape — Valley Crossing

### 4.1 Metastable Motion

**Cơ chế gốc (J Math Bio 2024):** Crossing rate ∝ K × μ^L. K=pop size, μ=mutation rate, L=valley width (số đột biến trung gian cần). Metastability graph: valley widths tạo time scales riêng — hẹp hơn → cross được, rộng hơn → stuck.

**Useful V2.9.x:**
- V2.9.1 valley confirmed: L ≥ 3 (đột biến "move hiệu quả" + "hướng food" + "ăn food" = 3 steps). Với K=1024, μ=0.1 → crossing rate ∝ 1.024 → rất thấp. Đây là lý do toán học valley không thể tự vượt.
- **Design tool:** Tính valley width từ số gen stuck → determine mutation rate cần.

**Ko useful:**
- Continuous-time Markov process proof — toán học quá sâu cho engineering
- General finite trait graph — env ant chỉ có 1 trait graph

---

### 4.2 Pit Stops

**Cơ chế gốc (arxiv 2025):** Intermediate mutants with POSITIVE fitness (even temporarily) act as pit stops. They grow to diverging size → survive longer → more chances for next mutation. Accelerates crossing by multiple time scales.

**Useful V2.9.2 VIP Init:**
- **VIP init = pit stop.** Teacher genome có fitness positive (~80) ngay gen 0 → cả pop khởi tạo từ đó. Valley width giảm từ L≥3 xuống L=0 (đã ở đỉnh).
- Nếu teacher ko perfect (fitness ~60 thay vì 80+) → vẫn là pit stop, accelerate crossing dù chưa hoàn toàn thoát valley.

**Ko useful:**
- Periodic environment (seasonal) detail — env ant constant
- Asexual reproduction mathematical model — V2.9.1 có crossover

---

### 4.3 Epistatic Hotspots

**Cơ chế gốc (PNAS 2025):** Sparse epistatic mutations BOOST evolvability. Reorient adaptive paths. Suboptimal peaks = stepping stones with migration (spatial structure in population).

**Useful V2.9.1:**
- Non-coding DNA + gene duplication = epistatic hotspots tự nhiên. Expression mutation (column 5) thay đổi mà ko ảnh hưởng active genes → tạo stepping stones.
- **Nếu valley quá rộng → tăng gene duplication rate** → tạo nhiều epistatic hotspots → tăng xác suất stepping stone.

**Ko useful:**
- Antibody protein landscape cụ thể — khác domain
- Experimental binding affinity data

---

### 4.4 Network Population Valley Crossing

**Cơ chế gốc (PMC 2024):** Population structure (graph topology) determines valley crossing. Low acceleration + low amplification → better valley crossing. Single-mutation amplifiers can both promote and suppress.

**Useful V2.9.1:**
- V2.9.1 pop structure = flat (mọi agent có thể crossover với mọi agent). Paper suggest low accel+low amplif tốt cho valley crossing → tournament selection (flat) ≈ optimal.
- **Ko cần thay đổi selection mechanism hiện tại.**

**Ko useful:**
- Graph theory metrics (amplification, acceleration) — áp dụng cho spatial pop, V2.9.1 pop flat
- Cancer evolution context — khác domain

---

## 5. Open-Ended Evolution

### 5.1 Novelty Search with Local Competition

**Cơ chế gốc (Lehman 2011-2025):** Abstract evolution as search for novel behaviors, not optimization. 3 variants: pure novelty (no objective), minimal criteria (threshold + novelty), NSLC (local competition + novelty). NSLC = diverse well-adapted solutions.

**Useful V2.9.x philosophy:**
- V2.9.1 gần với NSLC nhất trong literature:
  - GA mutation = novelty generation
  - Fitness = local competition (survival threshold)
  - Non-coding DNA = implicit novelty buffer
  - Dopamine gating = behavioral diversity
- **Design validated:** Ko cần thêm explicit novelty archive (MAP-Elites grid). Implicit mechanism đủ cho T4 compute.

**Ko useful:**
- Biped locomotion domain — khác env
- Virtual creature morphology — V2.9.x fixed body Ant

---

### 5.2 Dominated Novelty Search

**Cơ chế gốc (2025):** QD reformulated as GA with dynamic fitness transformations. No predefined bounds, no parameters. Drop-in replacement for MAP-Elites grid.

**Useful V2.9.x:**
- If V2.9.1 fitness landscape too flat (ko differentiation) → có thể dynamic fitness weighting. Nhưng hiện tại chưa cần — valley đã rõ.

**Ko useful:**
- QD benchmark domains — ko phải neuroevolution
- Structured/unstructured archive comparison — V2.9.1 ko dùng archive

---

### 5.3 ASAL

**Cơ chế gốc (2024):** Foundation models (vision-language) search for ALife simulations. Historical novelty in FM representation space. Discovers open-ended Lenia/Boids/GoL patterns.

**Ko useful — bỏ.**
- Foundation model inference quá nặng cho T4
- Simulation discovery (tìm cellular automata rules) — khác task với neuroevolution
- Không liên quan đến zero-reward survival

---

## 6. Synthesis

### Papers useful (áp dụng vào V2.9.x)

| Paper | Cơ chế | Genome param | Version |
|---|---|---|---|
| Grid cells — Object vectors (Nature Comms 2025) | Multi-slot memory cho food positions | `n_slots`, `slot_dim`, `recall_temp` | V2.9.3 |
| Grid cells — Trajectory (eLife 2025) | Path encoding thay vì position | `traj_len` | V2.9.3 |
| Theta sweeps (Nature 2025) | Multi-step mental simulation qua world model | `sweep_B`, `sweep_L`, `sweep_theta` | V2.9.3 |
| Place Vector-HaSH (Nature 2025) | Grid scaffold + place memory | `place_radius`, `place_lr`, `max_places` | V2.9.3 |
| Hebbian HC→MEC (PMC 2025) | Hebbian extension cho spatial memory | `hc_hebb_lr`, `anchor_salience` | V2.9.3 |
| Mirror Alignment (ICCV 2025) | Contrastive alignment obs↔action | `proj_dim`, `align_temp`, `align_lr` | V2.9.6 |
| GANE (GECCO 2023) | Adversarial imitation framework | `selectivity` | V2.9.6 |
| mPFC metacontrol (PNAS 2026) | Prediction error → controllability → adjust | `monitor_window`, `anomaly_threshold`, `reflect_interval` | V2.9.5 |
| Intrinsic Metacognition (PMLR 2025) | Philosophy: intrinsic vs extrinsic | Triết học cho toàn bộ V2.9.x | Philosophy |
| Fitness valley (J Math Bio 2024) | Valley width L → mutation rate design | Tính K·μ^L | V2.9.x tool |
| Pit stops (arxiv 2025) | VIP init = pit stop accelerator | VIP init genome | V2.9.2 |
| Epistatic hotspots (PNAS 2025) | Non-coding gene = stepping stone | Dup rate | V2.9.1 |

### Papers bỏ (ko useful cho zero-reward neuroevolution)

| Paper | Lý do bỏ |
|---|---|
| Dopamine default policy penalty (bioRxiv 2025) | RL-based reward, V2.9.1 zero reward |
| DA as RPE (Current Biology 2025) | RL-based, CEM stimulation |
| Tonic DA (Nature Comms 2025) | Reward learning bias |
| ACh gating (Sci Rep 2025) | RL-based three-factor rule |
| Robotic MNS UBAL (ICANN 2024) | Camera + visual pipeline, grasping-specific |
| ASAL (2024) | FM-based, simulation discovery, khác task |
| Meta-Dyna (Frontiers 2025) | Q-learning, tabula RL — nhưng phần arbitration useful cho V2.9.4 |
