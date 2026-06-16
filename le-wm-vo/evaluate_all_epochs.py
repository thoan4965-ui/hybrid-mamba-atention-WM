"""
Evaluate all epochs for CfC and AR models.
Compares performance across all checkpoints.
"""
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_model, predict_next_embedding


def build_indices(h5_path, num_samples=100, seed=42):
    """
    Build validation indices from H5 file.
    
    Returns:
        tuple: (indices list, frame metadata) for reuse across seeds
    """
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
        for start in range(0, max_start + 1):
            indices.append((ep_idx, start))
    
    np.random.seed(seed)
    sampled = np.random.choice(len(indices), min(num_samples, len(indices)), replace=False)
    sampled_indices = [indices[s] for s in sampled]
    
    return sampled_indices, ep_len, ep_offset


def load_validation_data(h5_path, sampled_indices, ep_len, ep_offset, device="cpu"):
    """
    Load validation data from H5 file using pre-computed indices.
    
    Args:
        h5_path: Path to dataset H5 file
        sampled_indices: List of (ep_idx, start) tuples
        device: "cpu" or "cuda"
    
    Returns:
        list of dicts with keys: pixels, actions
    """
    import h5py
    import hdf5plugin
    
    history_size = 3
    num_preds = 3
    frameskip = 3
    seq_len = history_size + num_preds
    span = seq_len * frameskip
    
    with h5py.File(h5_path, "r") as f:
        samples = []
        for idx in tqdm(sampled_indices, desc="Loading validation data"):
            ep_idx, start = idx
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


def evaluate_model(model_dict, val_data, device="cpu"):
    """
    Evaluate a model on validation data.
    
    Args:
        model_dict: Output from load_model()
        val_data: List of validation samples
        device: "cpu" or "cuda"
    
    Returns:
        dict with metrics: pred_loss, sigreg_loss, total_loss
    """
    encoder = model_dict["encoder"]
    predictor = model_dict["predictor"]
    action_encoder = model_dict["action_encoder"]
    sigreg = model_dict["sigreg"]
    model_type = model_dict["model_type"]
    
    # Parameters
    history_size = 3
    num_preds = 3
    lambd = 0.05  # sigreg weight
    
    pred_losses = []
    sigreg_losses = []
    
    for sample in tqdm(val_data, desc=f"Evaluating epoch {model_dict['epoch']}"):
        pixels = sample["pixels"].unsqueeze(0)  # (1, T, 3, H, W)
        actions = sample["actions"].unsqueeze(0)  # (1, T, action_dim)
        
        # Encode all pixels first
        B, T, C, H, W = pixels.shape
        pixels_flat = pixels.reshape(B * T, C, H, W)
        with torch.no_grad():
            all_pixel_emb = encoder(pixels_flat).reshape(B, T, -1)  # (1, 6, D)
        
        # Encode all actions
        with torch.no_grad():
            all_action_emb = action_encoder(actions)  # (1, 6, D)
        
        if model_type == "cfc_v3":
            # CfC V3: sequential RNN loop, ROLLOUT (feed own prediction)
            h = None
            # Phase 1: Build hidden state from history
            for t in range(history_size - 1):
                _, h = predictor.step(
                    all_pixel_emb[:, t:t+1], all_action_emb[:, t:t+1], h
                )
            # Phase 2: Rollout predict future frames
            predictions = []
            last_pred = None
            for t in range(num_preds):
                feed_idx = history_size - 1 + t
                feed_pix = all_pixel_emb[:, feed_idx:feed_idx+1] if last_pred is None else last_pred
                out, h = predictor.step(
                    feed_pix,
                    all_action_emb[:, feed_idx:feed_idx+1],
                    h,
                )
                predictions.append(out)
                last_pred = out
            pred_target = torch.cat(predictions, dim=1)
            target_emb = all_pixel_emb[:, history_size:]

        elif model_type == "cfc":
            # CfC V1/V2: receive 6 frames, predict last 3 frames
            pixel_emb = all_pixel_emb  # (1, 6, D)
            action_emb = all_action_emb  # (1, 6, D)
            
            with torch.no_grad():
                pred_emb = predictor(pixel_emb, action_emb)  # (1, 6, D)
            
            # Compare predicted frames 3,4,5 with actual frames 3,4,5
            target_emb = pixel_emb[:, history_size:]  # (1, 3, D)
            pred_target = pred_emb[:, history_size:]  # (1, 3, D)
            
        else:  # AR
            # AR: receive 3 frames (history), autoregressively predict 3 frames
            history_emb = all_pixel_emb[:, :history_size]  # (1, 3, D)
            history_act = all_action_emb[:, :history_size]  # (1, 3, D)
            
            predictions = []
            with torch.no_grad():
                for i in range(num_preds):
                    # Predict next frame using last 3 frames
                    pred = predictor(history_emb, history_act)[:, -1:]  # (1, 1, D)
                    predictions.append(pred)
                    
                    # Shift window: drop oldest, add newest
                    history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)  # (1, 3, D)
                    
                    # Add next action (frames 3, 4, 5)
                    if i < num_preds:
                        next_act = all_action_emb[:, history_size + i:history_size + i + 1]  # (1, 1, D)
                        history_act = torch.cat([history_act[:, 1:], next_act], dim=1)  # (1, 3, D)
            
            pred_target = torch.cat(predictions, dim=1)  # (1, 3, D)
            target_emb = all_pixel_emb[:, history_size:]  # (1, 3, D)
        
        pred_loss = (pred_target - target_emb).pow(2).mean().item()
        
        # Compute SIGReg loss on history embeddings
        with torch.no_grad():
            sigreg_input = all_pixel_emb[:, :history_size].transpose(0, 1)  # (3, 1, D)
            sigreg_loss = sigreg(sigreg_input).item()
        
        pred_losses.append(pred_loss)
        sigreg_losses.append(sigreg_loss)
    
    # Aggregate metrics
    avg_pred_loss = np.mean(pred_losses)
    avg_sigreg_loss = np.mean(sigreg_losses)
    avg_total_loss = avg_pred_loss + lambd * avg_sigreg_loss
    
    return {
        "pred_loss": avg_pred_loss,
        "sigreg_loss": avg_sigreg_loss,
        "total_loss": avg_total_loss,
    }


def evaluate_all_epochs(models_dir, model_type, h5_path, seeds=[42, 123, 456], num_samples=100, device="cpu"):
    """
    Evaluate all epochs for a given model type across multiple seeds.
    
    Args:
        models_dir: Directory containing checkpoints
        model_type: "cfc_v3", "cfc", or "ar"
        h5_path: Path to dataset H5 file
        seeds: List of random seeds for data sampling
        num_samples: Number of samples per seed
        device: "cpu" or "cuda"
    
    Returns:
        list of dicts with epoch and metrics (mean±std across seeds)
    """
    models_dir = Path(models_dir)
    glob_prefix = "vit_cfc_v3" if model_type == "cfc_v3" else f"vit_{model_type}"
    ckpt_files = sorted(models_dir.glob(f"{glob_prefix}_epoch_*.ckpt"))
    
    # Pre-load validation data for all seeds (cache to avoid H5 re-reads)
    print(f"  Pre-loading validation data for {len(seeds)} seeds...")
    val_data_by_seed = {}
    for seed in tqdm(seeds, desc=f"  Loading data seeds"):
        sampled_indices, ep_len, ep_offset = build_indices(h5_path, num_samples, seed)
        val_data_by_seed[seed] = load_validation_data(
            h5_path, sampled_indices, ep_len, ep_offset, device
        )
    
    results = []
    for ckpt_file in tqdm(ckpt_files, desc=f"Evaluating {model_type.upper()} models"):
        model_dict = load_model(str(ckpt_file), model_type=model_type, device=device)
        epoch = model_dict["epoch"]
        
        pred_losses = []
        sigreg_losses = []
        for seed in seeds:
            metrics = evaluate_model(model_dict, val_data_by_seed[seed], device=device)
            pred_losses.append(metrics["pred_loss"])
            sigreg_losses.append(metrics["sigreg_loss"])
        
        result = {
            "epoch": epoch,
            "model_type": model_type,
            "pred_loss": float(np.mean(pred_losses)),
            "pred_loss_std": float(np.std(pred_losses)),
            "sigreg_loss": float(np.mean(sigreg_losses)),
            "sigreg_loss_std": float(np.std(sigreg_losses)),
            "total_loss": float(np.mean(pred_losses) + 0.05 * np.mean(sigreg_losses)),
        }
        results.append(result)
        
        print(f"  Epoch {epoch}: pred_loss={result['pred_loss']:.6f} ± {result['pred_loss_std']:.6f}")
    
    return results


def main():
    seeds = [42, 123, 456]
    num_samples = 100
    
    # Paths
    models_dir = Path("D:/ai_training/MODELS")
    h5_path = Path("D:/ai_training/bionic_hand_dataset_v3_96.h5")
    
    # Evaluate CfC models
    print("=" * 80)
    print("Evaluating CfC V1/V2 models")
    print("=" * 80)
    cfc_results = evaluate_all_epochs(models_dir / "cfc_models", "cfc", h5_path, seeds, num_samples)
    
    # Evaluate CfC V3 models (in same dir as V1/V2)
    cfc_v3_dir = models_dir / "cfc_models"
    cfc_v3_results = []
    if cfc_v3_dir.exists() and list(cfc_v3_dir.glob("vit_cfc_v3_epoch_*.ckpt")):
        print("\n" + "=" * 80)
        print("Evaluating CfC V3 models")
        print("=" * 80)
        cfc_v3_results = evaluate_all_epochs(cfc_v3_dir, "cfc_v3", h5_path, seeds, num_samples)
    
    # Evaluate AR models
    print("\n" + "=" * 80)
    print("Evaluating AR models")
    print("=" * 80)
    ar_results = evaluate_all_epochs(models_dir / "ar_models", "ar", h5_path, seeds, num_samples)
    
    # Combine results
    all_results = cfc_results + cfc_v3_results + ar_results
    
    # Save results
    import json
    output_path = models_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Best CfC (using pred_loss mean, not total_loss)
    best_cfc = min(cfc_results, key=lambda x: x["pred_loss"])
    print(f"\nBest CfC: Epoch {best_cfc['epoch']}")
    print(f"  pred_loss: {best_cfc['pred_loss']:.6f} ± {best_cfc['pred_loss_std']:.6f}")
    
    if cfc_v3_results:
        best_cfc_v3 = min(cfc_v3_results, key=lambda x: x["pred_loss"])
        print(f"\nBest CfC V3: Epoch {best_cfc_v3['epoch']}")
        print(f"  pred_loss: {best_cfc_v3['pred_loss']:.6f} ± {best_cfc_v3['pred_loss_std']:.6f}")
    
    # Best AR
    best_ar = min(ar_results, key=lambda x: x["pred_loss"])
    print(f"\nBest AR: Epoch {best_ar['epoch']}")
    print(f"  pred_loss: {best_ar['pred_loss']:.6f} ± {best_ar['pred_loss_std']:.6f}")
    
    # Comparison (only CfC V3 vs AR if V3 exists, else CfC V1/V2)
    print("\n" + "=" * 80)
    print("COMPARISON (pred_loss)")
    print("=" * 80)
    cfcs_to_compare = cfc_v3_results if cfc_v3_results else cfc_results
    best_cfc = min(cfcs_to_compare, key=lambda x: x["pred_loss"])
    diff = best_cfc["pred_loss"] - best_ar["pred_loss"]
    if diff < 0:
        print(f"✓ CfC wins by {-diff:.6f} (mean pred_loss)")
    else:
        print(f"✓ AR wins by {diff:.6f} (mean pred_loss)")


if __name__ == "__main__":
    main()
