"""
Investigate models using REAL H5 data.
FIXED: CfC now receives only 3 history frames (matching training),
not all 6 frames (which allowed it to copy targets).
"""
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load_model


def load_real_samples(h5_path, num_samples=50, device="cpu"):
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

        np.random.seed(999)
        sampled_indices = np.random.choice(len(indices), min(num_samples, len(indices)), replace=False)

        samples = []
        for idx in sampled_indices:
            ep_idx, start = indices[idx]
            offset = ep_offset[ep_idx]

            px_idxs = [offset + start + i * frameskip for i in range(seq_len)]
            pixels = torch.from_numpy(f["pixels"][px_idxs]).permute(0, 3, 1, 2).float() / 255.0

            act_frames = []
            for i in range(seq_len):
                act_start = offset + start + i * frameskip
                act_chunk = torch.from_numpy(f["action"][act_start:act_start + frameskip])
                act_frames.append(act_chunk.reshape(-1))
            actions = torch.stack(act_frames)

            samples.append({
                "pixels": pixels.to(device),
                "actions": actions.to(device),
            })

    return samples


def investigate_cfc(model_dict, samples, device="cpu"):
    """CfC: feed only 3 history frames, predict 3 future frames (matching training)."""
    encoder = model_dict["encoder"]
    predictor = model_dict["predictor"]
    action_encoder = model_dict["action_encoder"]

    H = 3  # history_size

    normal_losses = []
    shuffled_losses = []
    zero_action_losses = []
    random_losses = []

    from module import CfCPredictorV2
    random_predictor = CfCPredictorV2(
        num_frames=3, input_dim=32, hidden_dim=64,
        output_dim=32, action_dim=32, backbone_layers=1, backbone_units=64,
    ).to(device)
    random_predictor.eval()

    for sample in tqdm(samples, desc="CfC investigation"):
        pixels = sample["pixels"].unsqueeze(0)  # (1, 6, 3, 96, 96)
        actions = sample["actions"].unsqueeze(0)  # (1, 6, 24)

        with torch.no_grad():
            # Encode ALL pixels and actions first
            all_pixel_emb = encoder(pixels.reshape(6, 3, 96, 96)).reshape(1, 6, -1)
            all_action_emb = action_encoder(actions)  # (1, 6, D)

            history_emb = all_pixel_emb[:, :H]  # (1, 3, D)
            history_act = all_action_emb[:, :H]   # (1, 3, D)
            target_emb = all_pixel_emb[:, H:]    # (1, 3, D) — actual future

            # === Test 1: Normal prediction (3 frames in, predict 3 future) ===
            pred = predictor(history_emb, history_act)  # (1, 3, D)
            normal_loss = (pred - target_emb).pow(2).mean().item()

            # === Test 2: Shuffle history frames ===
            perm = torch.randperm(H)
            shuffled_history = history_emb[:, perm, :]
            pred_shuffled = predictor(shuffled_history, history_act)
            shuffled_loss = (pred_shuffled - target_emb).pow(2).mean().item()

            # === Test 3: Zero actions ===
            zero_actions = torch.zeros_like(actions)
            zero_act_emb = action_encoder(zero_actions)
            history_act_zero = zero_act_emb[:, :H]
            pred_zero = predictor(history_emb, history_act_zero)
            zero_action_loss = (pred_zero - target_emb).pow(2).mean().item()

            # === Test 4: Random predictor ===
            pred_random = random_predictor(history_emb, history_act)
            random_loss = (pred_random - target_emb).pow(2).mean().item()

        normal_losses.append(normal_loss)
        shuffled_losses.append(shuffled_loss)
        zero_action_losses.append(zero_action_loss)
        random_losses.append(random_loss)

    return {
        "normal": np.mean(normal_losses),
        "shuffled": np.mean(shuffled_losses),
        "zero_action": np.mean(zero_action_losses),
        "random": np.mean(random_losses),
    }


def investigate_ar(model_dict, samples, device="cpu"):
    """AR: autoregressive rollout from 3 history frames."""
    encoder = model_dict["encoder"]
    predictor = model_dict["predictor"]
    action_encoder = model_dict["action_encoder"]

    H = 3
    num_preds = 3

    normal_losses = []
    shuffled_losses = []
    zero_action_losses = []
    random_losses = []

    from module import ARPredictor
    random_predictor = ARPredictor(
        num_frames=3, depth=1, heads=2, mlp_dim=96,
        input_dim=32, hidden_dim=64, output_dim=32, dim_head=16,
    ).to(device)
    random_predictor.eval()

    for sample in tqdm(samples, desc="AR investigation"):
        pixels = sample["pixels"].unsqueeze(0)
        actions = sample["actions"].unsqueeze(0)

        with torch.no_grad():
            all_pixel_emb = encoder(pixels.reshape(6, 3, 96, 96)).reshape(1, 6, -1)
            all_action_emb = action_encoder(actions)

            target_emb = all_pixel_emb[:, H:]  # (1, 3, D)

            # === Test 1: Normal AR rollout ===
            history_emb = all_pixel_emb[:, :H].clone()
            history_act = all_action_emb[:, :H].clone()
            predictions = []
            for i in range(num_preds):
                pred = predictor(history_emb, history_act)[:, -1:]
                predictions.append(pred)
                history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
                if i < num_preds - 1:
                    next_act = all_action_emb[:, H + i:H + i + 1]
                    history_act = torch.cat([history_act[:, 1:], next_act], dim=1)
            pred_target = torch.cat(predictions, dim=1)
            normal_loss = (pred_target - target_emb).pow(2).mean().item()

            # === Test 2: Shuffle history frames ===
            perm = torch.randperm(H)
            shuffled_history = all_pixel_emb[:, :H][:, perm, :].clone()
            history_act = all_action_emb[:, :H].clone()
            predictions = []
            for i in range(num_preds):
                pred = predictor(shuffled_history, history_act)[:, -1:]
                predictions.append(pred)
                shuffled_history = torch.cat([shuffled_history[:, 1:], pred], dim=1)
                if i < num_preds - 1:
                    next_act = all_action_emb[:, H + i:H + i + 1]
                    history_act = torch.cat([history_act[:, 1:], next_act], dim=1)
            pred_target = torch.cat(predictions, dim=1)
            shuffled_loss = (pred_target - target_emb).pow(2).mean().item()

            # === Test 3: Zero actions ===
            history_emb = all_pixel_emb[:, :H].clone()
            zero_actions = torch.zeros_like(actions)
            zero_act_emb = action_encoder(zero_actions)
            history_act_zero = zero_act_emb[:, :H].clone()
            predictions = []
            for i in range(num_preds):
                pred = predictor(history_emb, history_act_zero)[:, -1:]
                predictions.append(pred)
                history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
                if i < num_preds - 1:
                    next_act = zero_act_emb[:, H + i:H + i + 1]
                    history_act_zero = torch.cat([history_act_zero[:, 1:], next_act], dim=1)
            pred_target = torch.cat(predictions, dim=1)
            zero_action_loss = (pred_target - target_emb).pow(2).mean().item()

            # === Test 4: Random predictor ===
            history_emb = all_pixel_emb[:, :H].clone()
            history_act = all_action_emb[:, :H].clone()
            predictions = []
            for i in range(num_preds):
                pred = random_predictor(history_emb, history_act)[:, -1:]
                predictions.append(pred)
                history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
                if i < num_preds - 1:
                    next_act = all_action_emb[:, H + i:H + i + 1]
                    history_act = torch.cat([history_act[:, 1:], next_act], dim=1)
            pred_target = torch.cat(predictions, dim=1)
            random_loss = (pred_target - target_emb).pow(2).mean().item()

        normal_losses.append(normal_loss)
        shuffled_losses.append(shuffled_loss)
        zero_action_losses.append(zero_action_loss)
        random_losses.append(random_loss)

    return {
        "normal": np.mean(normal_losses),
        "shuffled": np.mean(shuffled_losses),
        "zero_action": np.mean(zero_action_losses),
        "random": np.mean(random_losses),
    }


def print_results(name, results):
    normal = results["normal"]
    shuffled = results["shuffled"]
    zero_action = results["zero_action"]
    random_ = results["random"]

    print(f"\n{'='*80}")
    print(f"  {name} - Investigation Results (REAL DATA, FIXED)")
    print(f"{'='*80}")
    print(f"  Normal prediction:    {normal:.6f}")
    print(f"  Shuffled input:       {shuffled:.6f}  ({shuffled/normal:.2f}x)")
    print(f"  Zero actions:         {zero_action:.6f}  ({zero_action/normal:.2f}x)")
    print(f"  Random predictor:    {random_:.6f}  ({random_/normal:.2f}x)")

    issues = []
    passes = []
    if shuffled / normal < 1.5:
        issues.append(f"FAIL: Shuffled only {shuffled/normal:.2f}x worse -> copying input")
    elif shuffled / normal < 2.0:
        issues.append(f"WEAK: Shuffled only {shuffled/normal:.2f}x worse -> partial temporal")
    else:
        passes.append(f"PASS: Temporal dynamics ({shuffled/normal:.2f}x)")

    if zero_action / normal < 1.5:
        issues.append(f"FAIL: Zero-action only {zero_action/normal:.2f}x worse -> ignoring actions")
    elif zero_action / normal < 2.0:
        issues.append(f"WEAK: Zero-action only {zero_action/normal:.2f}x worse -> weak action use")
    else:
        passes.append(f"PASS: Action conditioning ({zero_action/normal:.2f}x)")

    if random_ / normal < 2.0:
        issues.append(f"FAIL: Random only {random_/normal:.2f}x worse -> not better than random")
    elif random_ / normal < 5.0:
        issues.append(f"WEAK: Random only {random_/normal:.2f}x worse -> marginally better")
    else:
        passes.append(f"PASS: Better than random ({random_/normal:.2f}x)")

    for p in passes:
        print(f"  {p}")
    if issues:
        print(f"  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ALL TESTS PASSED")

    return len(issues) == 0


def main():
    device = "cpu"
    h5_path = Path("D:/ai_training/bionic_hand_dataset_v3_96.h5")

    print("Loading real H5 data...")
    samples = load_real_samples(h5_path, num_samples=50, device=device)
    print(f"Loaded {len(samples)} real samples\n")

    # Test CfC (CORRECTED: only 3 history frames as input)
    print("Loading CfC epoch 10...")
    cfc_model = load_model("D:/ai_training/MODELS/cfc_models/vit_cfc_epoch_10.ckpt",
                           model_type="cfc", device=device)
    cfc_results = investigate_cfc(cfc_model, samples, device)
    cfc_pass = print_results("CfC", cfc_results)

    # Test AR (same as before - already correct)
    print("\nLoading AR epoch 10...")
    ar_model = load_model("D:/ai_training/MODELS/ar_models/vit_ar_epoch_10.ckpt",
                          model_type="ar", device=device)
    ar_results = investigate_ar(ar_model, samples, device)
    ar_pass = print_results("AR", ar_results)

    # Also test best epochs
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if cfc_pass:
        print("CfC PASSED investigation - viable for real robot")
    else:
        print("CfC FAILED investigation")

    if ar_pass:
        print("AR PASSED investigation - viable for real robot")
    else:
        print("AR FAILED investigation")

    if not cfc_pass and not ar_pass:
        print("\nNeither model passed - consider:")
        print("  1. Retrain with more epochs or different hyperparameters")
        print("  2. Add explicit action-conditioning loss")
        print("  3. Use larger frameskip for harder prediction task")


if __name__ == "__main__":
    main()