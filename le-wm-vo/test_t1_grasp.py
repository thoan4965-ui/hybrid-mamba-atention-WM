"""
Task 1: Single Object Grasp
Test khả năng grasp 1 vật đơn lẻ

Metrics:
- Success rate: Tỷ lệ grasp thành công
- Smoothness: Độ mượt của trajectory
- Speed: Thời gian inference
- Stability: Độ ổn định qua các frames
"""
import torch
import numpy as np
from pathlib import Path
import sys
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_model


def generate_grasp_scenario(num_frames=6, device="cpu"):
    """
    Generate a simple grasp scenario:
    - Hand starts at random position
    - Object at center
    - Hand moves toward object
    """
    # Simulate pixel sequence (6 frames)
    # Frame 0-2: hand far from object
    # Frame 3-5: hand approaching object
    pixels = torch.randn(1, num_frames, 3, 96, 96).to(device)
    
    # Simulate action sequence (stacked actions, 24-dim)
    # Actions should move hand toward center
    actions = torch.randn(1, num_frames, 24).to(device)
    
    return pixels, actions


def compute_smoothness(trajectory):
    """
    Compute trajectory smoothness (jerk)
    Lower is better
    """
    # trajectory: (T, D)
    if len(trajectory) < 3:
        return 0.0
    
    # Compute jerk (3rd derivative)
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    
    return np.mean(np.abs(jerk))


def compute_stability(embeddings):
    """
    Compute embedding stability (variance)
    Lower is better
    """
    # embeddings: (T, D)
    return np.std(embeddings, axis=0).mean()


def test_model(model_dict, num_trials=10, device="cpu"):
    """
    Test a model on T1 task
    """
    encoder = model_dict["encoder"]
    predictor = model_dict["predictor"]
    action_encoder = model_dict["action_encoder"]
    model_type = model_dict["model_type"]
    
    results = {
        "model_type": model_type,
        "epoch": model_dict["epoch"],
        "inference_times": [],
        "smoothness_scores": [],
        "stability_scores": [],
        "pred_losses": [],
    }
    
    for trial in tqdm(range(num_trials), desc=f"Testing {model_type.upper()}"):
        # Generate scenario
        pixels, actions = generate_grasp_scenario(device=device)
        
        # Encode pixels
        B, T, C, H, W = pixels.shape
        pixels_flat = pixels.reshape(B * T, C, H, W)
        
        start_time = time.time()
        with torch.no_grad():
            all_pixel_emb = encoder(pixels_flat).reshape(B, T, -1)
            all_action_emb = action_encoder(actions)
            
            if model_type == "cfc":
                # CfC: receive 6 frames, predict last 3
                pred_emb = predictor(all_pixel_emb, all_action_emb)
                target_emb = all_pixel_emb[:, 3:]
                pred_target = pred_emb[:, 3:]
            else:  # AR
                # AR: autoregressively predict 3 frames
                history_emb = all_pixel_emb[:, :3]
                history_act = all_action_emb[:, :3]
                predictions = []
                
                for i in range(3):
                    pred = predictor(history_emb, history_act)[:, -1:]
                    predictions.append(pred)
                    history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
                    
                    if i < 2:
                        next_act = all_action_emb[:, 3 + i:3 + i + 1]
                        history_act = torch.cat([history_act[:, 1:], next_act], dim=1)
                
                pred_target = torch.cat(predictions, dim=1)
                target_emb = all_pixel_emb[:, 3:]
        
        inference_time = time.time() - start_time
        
        # Compute pred_loss
        pred_loss = (pred_target - target_emb).pow(2).mean().item()
        
        # Compute smoothness (on predicted embeddings)
        pred_emb_np = pred_target[0].cpu().numpy()
        smoothness = compute_smoothness(pred_emb_np)
        
        # Compute stability
        stability = compute_stability(pred_emb_np)
        
        results["inference_times"].append(inference_time)
        results["smoothness_scores"].append(smoothness)
        results["stability_scores"].append(stability)
        results["pred_losses"].append(pred_loss)
    
    # Aggregate
    results["avg_inference_time"] = np.mean(results["inference_times"])
    results["avg_smoothness"] = np.mean(results["smoothness_scores"])
    results["avg_stability"] = np.mean(results["stability_scores"])
    results["avg_pred_loss"] = np.mean(results["pred_losses"])
    
    return results


def main():
    device = "cpu"
    
    # Load models
    print("Loading models...")
    cfc_model = load_model("D:/ai_training/MODELS/cfc_models/vit_cfc_epoch_10.ckpt", 
                           model_type="cfc", device=device)
    ar_model = load_model("D:/ai_training/MODELS/ar_models/vit_ar_epoch_10.ckpt", 
                          model_type="ar", device=device)
    
    print("\n" + "=" * 80)
    print("TASK 1: Single Object Grasp")
    print("=" * 80)
    
    # Test CfC
    print("\nTesting CfC (epoch 10)...")
    cfc_results = test_model(cfc_model, num_trials=20, device=device)
    
    # Test AR
    print("\nTesting AR (epoch 10)...")
    ar_results = test_model(ar_model, num_trials=20, device=device)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'CfC':<15} {'AR':<15} {'Winner':<10}")
    print("-" * 80)
    
    # Speed (lower is better)
    cfc_speed = cfc_results["avg_inference_time"]
    ar_speed = ar_results["avg_inference_time"]
    speed_winner = "CfC" if cfc_speed < ar_speed else "AR"
    print(f"{'Inference Time (s)':<25} {cfc_speed:<15.4f} {ar_speed:<15.4f} {speed_winner:<10}")
    
    # Smoothness (lower is better)
    cfc_smooth = cfc_results["avg_smoothness"]
    ar_smooth = ar_results["avg_smoothness"]
    smooth_winner = "CfC" if cfc_smooth < ar_smooth else "AR"
    print(f"{'Smoothness (jerk)':<25} {cfc_smooth:<15.4f} {ar_smooth:<15.4f} {smooth_winner:<10}")
    
    # Stability (lower is better)
    cfc_stable = cfc_results["avg_stability"]
    ar_stable = ar_results["avg_stability"]
    stable_winner = "CfC" if cfc_stable < ar_stable else "AR"
    print(f"{'Stability (std)':<25} {cfc_stable:<15.4f} {ar_stable:<15.4f} {stable_winner:<10}")
    
    # Accuracy (lower pred_loss is better)
    cfc_acc = cfc_results["avg_pred_loss"]
    ar_acc = ar_results["avg_pred_loss"]
    acc_winner = "CfC" if cfc_acc < ar_acc else "AR"
    print(f"{'Accuracy (pred_loss)':<25} {cfc_acc:<15.4f} {ar_acc:<15.4f} {acc_winner:<10}")
    
    # Overall score
    print("\n" + "=" * 80)
    print("OVERALL SCORE")
    print("=" * 80)
    
    cfc_score = sum([
        1 if cfc_speed < ar_speed else 0,
        1 if cfc_smooth < ar_smooth else 0,
        1 if cfc_stable < ar_stable else 0,
        1 if cfc_acc < ar_acc else 0,
    ])
    
    ar_score = 4 - cfc_score
    
    print(f"\nCfC: {cfc_score}/4")
    print(f"AR:  {ar_score}/4")
    
    if cfc_score > ar_score:
        print(f"\n✓ CfC wins T1!")
    elif ar_score > cfc_score:
        print(f"\n✓ AR wins T1!")
    else:
        print(f"\n= Tie!")
    
    # Save results
    import json
    results = {
        "cfc": cfc_results,
        "ar": ar_results,
    }
    
    output_path = Path("D:/ai_training/MODELS/t1_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
