"""
Fair evaluation: CfC vs AR — cả hai nhận 3 context frames, predict 3 future frames.

CfC: sequential RNN, carry hidden state
AR: autoregressive transformer, shift context window

Tests:
1. Normal prediction (MSE)
2. Shuffle test (temporal dynamics)
3. Zero-action test (action conditioning)
4. Random predictor baseline
5. Speed test (inference time)
"""
import torch
import numpy as np
import time
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load_model


def build_fair_indices(h5_path, num_samples=50, seed=999):
    import h5py
    import hdf5plugin

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

    history_size = 3
    num_preds = 3
    frameskip = 3
    seq_len = history_size + num_preds
    span = seq_len * frameskip

    indices = []
    for ep_idx in range(len(ep_len)):
        max_start = ep_len[ep_idx] - span
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, 3):
            indices.append((ep_idx, start))

    np.random.seed(seed)
    sampled = np.random.choice(len(indices), min(num_samples, len(indices)), replace=False)
    return [indices[s] for s in sampled], ep_len, ep_offset


def load_real_samples(h5_path, sampled_indices, ep_len, ep_offset, device="cpu"):
    import h5py
    import hdf5plugin

    history_size = 3
    num_preds = 3
    frameskip = 3
    seq_len = history_size + num_preds
    span = seq_len * frameskip

    with h5py.File(h5_path, "r") as f:
        raw_action = torch.from_numpy(f["action"][:])
        raw_pixels = torch.from_numpy(f["pixels"][:])

        samples = []
        for idx in sampled_indices:
            ep_idx, start = idx
            offset = ep_offset[ep_idx]

            px_idxs = [offset + start + i * frameskip for i in range(seq_len)]
            pixels = raw_pixels[px_idxs].permute(0, 3, 1, 2).float() / 255.0

            act_frames = []
            for i in range(seq_len):
                act_start = offset + start + i * frameskip
                act_chunk = raw_action[act_start:act_start + frameskip]
                act_frames.append(act_chunk.reshape(-1))
            actions = torch.stack(act_frames)

            cfc_idxs = [offset + start + i * frameskip for i in range(seq_len)]
            actions_cfc = raw_action[cfc_idxs]

            samples.append({
                "pixels": pixels.to(device),
                "actions": actions.to(device),
                "actions_cfc": actions_cfc.to(device),
            })

    return samples


def predict_cfc(model_dict, pixel_emb, action_emb, history_size=3, num_preds=3):
    """CfC V1/V2: batch-style — processes entire history at once."""
    predictor = model_dict["predictor"]

    history_emb = pixel_emb[:, :history_size]
    history_act = action_emb[:, :history_size]
    target_emb = pixel_emb[:, history_size:history_size + num_preds]

    pred_emb = predictor(history_emb, history_act)
    pred_target = pred_emb[:, -num_preds:]

    return pred_target, target_emb


def predict_cfc_v3(model_dict, pixel_emb, action_emb, history_size=3, num_preds=3):
    """CfC V3: sequential RNN loop, ROLLOUT (feed own prediction)."""
    predictor = model_dict["predictor"]

    target_emb = pixel_emb[:, history_size:history_size + num_preds]

    # Phase 1: Build hidden state from history (no loss)
    h = None
    for t in range(history_size - 1):
        _, h = predictor.step(
            pixel_emb[:, t:t+1], action_emb[:, t:t+1], h
        )

    # Phase 2: Rollout predict future frames
    predictions = []
    last_pred = None
    for t in range(num_preds):
        feed_idx = history_size - 1 + t
        feed = pixel_emb[:, feed_idx:feed_idx+1] if last_pred is None else last_pred
        out, h = predictor.step(
            feed,
            action_emb[:, feed_idx:feed_idx+1],
            h,
        )
        predictions.append(out)
        last_pred = out

    pred_target = torch.cat(predictions, dim=1)
    return pred_target, target_emb


def predict_ar(model_dict, pixel_emb, action_emb, history_size=3, num_preds=3):
    """AR: autoregressive rollout from 3 context frames."""
    predictor = model_dict["predictor"]

    history_emb = pixel_emb[:, :history_size].clone()
    history_act = action_emb[:, :history_size].clone()
    target_emb = pixel_emb[:, history_size:history_size + num_preds]
    future_act = action_emb[:, history_size:history_size + num_preds]

    predictions = []
    for i in range(num_preds):
        pred = predictor(history_emb, history_act)[:, -1:]
        predictions.append(pred)
        history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
        if i < num_preds - 1:
            next_act = future_act[:, i:i + 1]
            history_act = torch.cat([history_act[:, 1:], next_act], dim=1)

    pred_target = torch.cat(predictions, dim=1)  # (1, 3, D)
    return pred_target, target_emb


def evaluate_fair(model_dict, samples, model_type, device="cpu"):
    """Fair evaluation: 3 context frames, predict 3 future frames."""
    encoder = model_dict["encoder"]
    action_encoder = model_dict["action_encoder"]
    predictor = model_dict["predictor"]

    history_size = 3
    num_preds = 3

    normal_losses = []
    shuffled_losses = []
    zero_action_losses = []
    random_losses = []

    # Random predictor baseline
    if model_type == "cfc_v3":
        from module import CfCPredictorV2
        random_predictor = CfCPredictorV2(
            num_frames=6, input_dim=32, hidden_dim=128,
            output_dim=32, action_dim=32, backbone_layers=2, backbone_units=128,
        ).to(device)
    elif model_type == "cfc_v2":
        from module import CfCPredictorV2
        random_predictor = CfCPredictorV2(
            num_frames=3, input_dim=32, hidden_dim=96,
            output_dim=32, action_dim=32, backbone_layers=1, backbone_units=96,
        ).to(device)
    elif model_type == "cfc":
        from module import CfCPredictorV2
        random_predictor = CfCPredictorV2(
            num_frames=3, input_dim=32, hidden_dim=128,
            output_dim=32, action_dim=32, backbone_layers=2, backbone_units=128,
        ).to(device)
    else:
        from module import ARPredictor
        random_predictor = ARPredictor(
            num_frames=3, depth=1, heads=2, mlp_dim=96,
            input_dim=32, hidden_dim=64, output_dim=32, dim_head=16,
        ).to(device)
    random_predictor.eval()

    random_model = {
        "encoder": encoder,
        "predictor": random_predictor,
        "action_encoder": action_encoder,
        "model_type": model_type,
    }

    if model_type == "cfc_v3":
        pred_fn = predict_cfc_v3
    elif model_type in ("cfc", "cfc_v2"):
        pred_fn = predict_cfc
    else:
        pred_fn = predict_ar

    action_key = "actions_cfc" if model_type in ("cfc_v3", "cfc_v2") else "actions"

    for sample in tqdm(samples, desc=f"{model_type.upper()} fair eval"):
        pixels = sample["pixels"].unsqueeze(0)
        actions = sample[action_key].unsqueeze(0)

        with torch.no_grad():
            all_pixel_emb = encoder(pixels.reshape(6, 3, 96, 96)).reshape(1, 6, -1)
            all_action_emb = action_encoder(actions)

            # === Test 1: Normal prediction ===
            pred_target, target_emb = pred_fn(model_dict, all_pixel_emb, all_action_emb)
            normal_loss = (pred_target - target_emb).pow(2).mean().item()

            # === Test 2: Shuffle history ===
            perm = torch.randperm(history_size)
            shuffled_emb = all_pixel_emb.clone()
            shuffled_emb[:, :history_size] = all_pixel_emb[:, perm]
            pred_shuf, _ = pred_fn(model_dict, shuffled_emb, all_action_emb)
            shuffled_loss = (pred_shuf - target_emb).pow(2).mean().item()

            # === Test 3: Zero actions ===
            zero_actions = torch.zeros_like(actions)
            zero_act_emb = action_encoder(zero_actions)
            zero_emb = all_pixel_emb.clone()
            pred_zero, _ = pred_fn(model_dict, zero_emb, zero_act_emb)
            zero_action_loss = (pred_zero - target_emb).pow(2).mean().item()

            # === Test 4: Random baseline ===
            pred_rand, _ = pred_fn(random_model, all_pixel_emb, all_action_emb)
            random_loss = (pred_rand - target_emb).pow(2).mean().item()

        normal_losses.append(normal_loss)
        shuffled_losses.append(shuffled_loss)
        zero_action_losses.append(zero_action_loss)
        random_losses.append(random_loss)

    # Speed test
    sample = samples[0]
    pixels = sample["pixels"].unsqueeze(0)
    actions = sample[action_key].unsqueeze(0)

    with torch.no_grad():
        all_pixel_emb = encoder(pixels.reshape(6, 3, 96, 96)).reshape(1, 6, -1)
        all_action_emb = action_encoder(actions)

        # Warmup
        for _ in range(10):
            pred_fn(model_dict, all_pixel_emb, all_action_emb)

        # Time it
        times = []
        for _ in range(100):
            start = time.perf_counter()
            pred_fn(model_dict, all_pixel_emb, all_action_emb)
            times.append(time.perf_counter() - start)

    avg_time_ms = np.mean(times) * 1000

    return {
        "normal": np.mean(normal_losses),
        "shuffled": np.mean(shuffled_losses),
        "zero_action": np.mean(zero_action_losses),
        "random": np.mean(random_losses),
        "speed_ms": avg_time_ms,
    }


def print_comparison(cfc_results, ar_results, cfc_label="CfC"):
    print(f"\n{'='*80}")
    print(f"  FAIR COMPARISON: {cfc_label} vs AR (3 context → predict 3 future)")
    print(f"  Multi-seed: mean±std across 3 seeds")
    print(f"{'='*80}")
    print(f"  {'Metric':<25} {cfc_label:>18} {'AR':>18} {'Ratio':>10}")
    print(f"  {'-'*69}")

    for key in ["normal", "shuffled", "zero_action", "random"]:
        c_mean = cfc_results[key]
        c_std = cfc_results.get(f"{key}_std", 0)
        a_mean = ar_results[key]
        a_std = ar_results.get(f"{key}_std", 0)
        ratio = c_mean / a_mean if a_mean != 0 else float('inf')
        label = {"normal": "Pred Loss", "shuffled": "Shuffled", "zero_action": "Zero-Action", "random": "Random"}[key]
        print(f"  {label:<25} {c_mean:>10.6f}±{c_std:.6f} {a_mean:>10.6f}±{a_std:.6f} {ratio:>10.2f}x")

    print(f"\n  {'Speed (ms)':<25} {cfc_results['speed_ms']:>12.2f}   {ar_results['speed_ms']:>12.2f}")

    cfc_shuffle_ratio = cfc_results["shuffled"] / cfc_results["normal"] if cfc_results["normal"] != 0 else 0
    ar_shuffle_ratio = ar_results["shuffled"] / ar_results["normal"] if ar_results["normal"] != 0 else 0
    cfc_action_ratio = cfc_results["zero_action"] / cfc_results["normal"] if cfc_results["normal"] != 0 else 0
    ar_action_ratio = ar_results["zero_action"] / ar_results["normal"] if ar_results["normal"] != 0 else 0
    cfc_vs_random = cfc_results["random"] / cfc_results["normal"] if cfc_results["normal"] != 0 else 0
    ar_vs_random = ar_results["random"] / ar_results["normal"] if ar_results["normal"] != 0 else 0

    print(f"\n  {'='*69}")
    print(f"  QUALITATIVE METRICS:")
    print(f"  {'Metric':<25} {cfc_label:>18} {'AR':>18}")
    print(f"  {'-'*69}")
    print(f"  {'Shuffle/Normal':<25} {cfc_shuffle_ratio:>18.2f}x {ar_shuffle_ratio:>18.2f}x")
    print(f"  {'ZeroAct/Normal':<25} {cfc_action_ratio:>18.2f}x {ar_action_ratio:>18.2f}x")
    print(f"  {'Random/Normal':<25} {cfc_vs_random:>18.2f}x {ar_vs_random:>18.2f}x")

    print(f"\n  VERDICT:")
    verdicts = []
    if cfc_shuffle_ratio > ar_shuffle_ratio:
        verdicts.append("CfC better at temporal dynamics")
    else:
        verdicts.append("AR better at temporal dynamics")

    if cfc_action_ratio > ar_action_ratio:
        verdicts.append("CfC better at action conditioning")
    else:
        verdicts.append("AR better at action conditioning")

    if cfc_vs_random > ar_vs_random:
        verdicts.append("CfC more above random baseline")
    else:
        verdicts.append("AR more above random baseline")

    if cfc_results["speed_ms"] < ar_results["speed_ms"]:
        verdicts.append(f"CfC faster ({cfc_results['speed_ms']:.1f}ms vs {ar_results['speed_ms']:.1f}ms)")
    else:
        verdicts.append(f"AR faster ({ar_results['speed_ms']:.1f}ms vs {cfc_results['speed_ms']:.1f}ms)")

    for v in verdicts:
        print(f"    → {v}")


def evaluate_model_multi_seed(ckpt_file, model_type, h5_path, seeds, num_samples, device, val_cache):
    """Evaluate a single checkpoint across multiple seeds. Returns dict of mean±std metrics."""
    model_dict = load_model(str(ckpt_file), model_type=model_type, device=device)

    all_metrics = []
    for seed in seeds:
        samples = load_real_samples(
            h5_path, val_cache[seed]["indices"], val_cache[seed]["ep_len"],
            val_cache[seed]["ep_offset"], device
        )
        metrics = evaluate_fair(model_dict, samples, model_type, device)
        all_metrics.append(metrics)

    result = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        result[key] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))
    return result


def find_best_model(models_dir, model_type, h5_path, seeds, num_samples, device, val_cache):
    """Find best checkpoint by mean normal loss across seeds."""
    models_dir = Path(models_dir)
    glob_prefix = f"vit_{model_type}"
    ckpt_files = sorted(models_dir.glob(f"{glob_prefix}_epoch_*.ckpt"))

    best_epoch = None
    best_results = None
    best_loss = float("inf")

    for ckpt_file in tqdm(ckpt_files, desc=f"Evaluating {model_type.upper()} epochs"):
        try:
            results = evaluate_model_multi_seed(
                ckpt_file, model_type, h5_path, seeds, num_samples, device, val_cache
            )
            loss = results["normal"]
            epoch = int(ckpt_file.stem.split("_")[-1])
            print(f"  Epoch {epoch}: normal={loss:.6f} ± {results['normal_std']:.6f}")
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_results = results
        except Exception as e:
            print(f"  {ckpt_file.name}: FAILED - {e}")

    print(f"✓ Best {model_type}: epoch {best_epoch} (normal={best_loss:.6f})")
    return best_epoch, best_results


def main():
    device = "cpu"
    h5_path = Path("D:/ai_training/bionic_hand_dataset_v3_96.h5")
    seeds = [42, 123, 456]
    num_samples = 50

    # Pre-build indices for all seeds (avoid H5 metadata re-reads)
    print("Building indices for all seeds...")
    val_cache = {}
    for seed in seeds:
        indices, ep_len, ep_offset = build_fair_indices(h5_path, num_samples, seed)
        val_cache[seed] = {"indices": indices, "ep_len": ep_len, "ep_offset": ep_offset}

    # Find best CfC V2
    print("\n" + "=" * 80)
    print("Finding best CfC V2")
    print("=" * 80)
    cfc_best_epoch, cfc_results = find_best_model(
        Path("D:/ai_training/MODELS/cfc_models"), "cfc_v2",
        h5_path, seeds, num_samples, device, val_cache
    )

    # Find best CfC V3 (in cfc_models dir)
    cfc_v3_best_epoch, cfc_v3_results = None, None
    cfc_v3_dir = Path("D:/ai_training/MODELS/cfc_models")
    if cfc_v3_dir.exists() and list(cfc_v3_dir.glob("vit_cfc_v3_epoch_*.ckpt")):
        print("\n" + "=" * 80)
        print("Finding best CfC V3")
        print("=" * 80)
        cfc_v3_best_epoch, cfc_v3_results = find_best_model(
            cfc_v3_dir, "cfc_v3",
            h5_path, seeds, num_samples, device, val_cache
        )

    # Find best AR
    print("\n" + "=" * 80)
    print("Finding best AR")
    print("=" * 80)
    ar_best_epoch, ar_results = find_best_model(
        Path("D:/ai_training/MODELS/ar_models"), "ar",
        h5_path, seeds, num_samples, device, val_cache
    )

    # Compare best CfC (V3 if exists else V2) vs best AR
    best_cfc_label = "CfC V3" if cfc_v3_results else "CfC V2"
    best_cfc_results = cfc_v3_results if cfc_v3_results else cfc_results
    print("\n" + "=" * 80)
    print(f"COMPARISON: Best {best_cfc_label} (epoch {cfc_v3_best_epoch if cfc_v3_results else cfc_best_epoch}) vs Best AR (epoch {ar_best_epoch})")
    print("=" * 80)
    print_comparison(best_cfc_results, ar_results, cfc_label=best_cfc_label)

    # Save results
    import json
    results = {
        "cfc_best_epoch": cfc_v3_best_epoch if cfc_v3_results else cfc_best_epoch,
        "cfc_label": best_cfc_label,
        "cfc": best_cfc_results,
        "ar_best_epoch": ar_best_epoch,
        "ar": ar_results,
    }
    output_path = Path("D:/ai_training/MODELS/fair_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()