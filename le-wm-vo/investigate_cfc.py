"""
Investigate CfC model - check if it's actually learning
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load_model


def test_cfc_investigation():
    device = "cpu"
    
    # Load CfC epoch 10
    print("Loading CfC epoch 10...")
    cfc_model = load_model("D:/ai_training/MODELS/cfc_models/vit_cfc_epoch_10.ckpt",
                           model_type="cfc", device=device)
    
    encoder = cfc_model["encoder"]
    predictor = cfc_model["predictor"]
    action_encoder = cfc_model["action_encoder"]
    
    print("\n" + "="*80)
    print("INVESTIGATION TESTS")
    print("="*80)
    
    # Test 1: Normal prediction
    print("\n[Test 1] Normal prediction (baseline)")
    pixels = torch.randn(1, 6, 3, 96, 96).to(device)
    actions = torch.randn(1, 6, 24).to(device)
    
    B, T, C, H, W = pixels.shape
    pixels_flat = pixels.reshape(B * T, C, H, W)
    
    with torch.no_grad():
        all_pixel_emb = encoder(pixels_flat).reshape(B, T, -1)
        all_action_emb = action_encoder(actions)
        
        # CfC: predict all 6 frames
        pred_emb = predictor(all_pixel_emb, all_action_emb)
        target_emb = all_pixel_emb[:, 3:]  # Compare with frames 3-5
        pred_target = pred_emb[:, 3:]
        
        normal_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {normal_loss:.6f}")
    
    # Test 2: Shuffle input frames
    print("\n[Test 2] Shuffle input frames (check if copying)")
    shuffled_pixels = pixels[:, torch.randperm(T)]  # Shuffle time dimension
    shuffled_pixels_flat = shuffled_pixels.reshape(B * T, C, H, W)
    
    with torch.no_grad():
        shuffled_pixel_emb = encoder(shuffled_pixels_flat).reshape(B, T, -1)
        
        pred_emb = predictor(shuffled_pixel_emb, all_action_emb)
        pred_target = pred_emb[:, 3:]
        target_emb = all_pixel_emb[:, 3:]  # Still compare with original target
        
        shuffled_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {shuffled_loss:.6f}")
    print(f"  Ratio (shuffled/normal): {shuffled_loss/normal_loss:.2f}x")
    
    if shuffled_loss / normal_loss < 2.0:
        print("  ⚠️  WARNING: CfC still predicts well with shuffled input!")
        print("     → Likely copying input instead of learning dynamics")
    else:
        print("  ✓ Good: CfC struggles with shuffled input")
        print("     → Actually learning temporal dynamics")
    
    # Test 3: Zero actions
    print("\n[Test 3] Zero actions (check if using actions)")
    zero_actions = torch.zeros_like(actions)
    zero_action_emb = action_encoder(zero_actions)
    
    with torch.no_grad():
        pred_emb = predictor(all_pixel_emb, zero_action_emb)
        pred_target = pred_emb[:, 3:]
        target_emb = all_pixel_emb[:, 3:]
        
        zero_action_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {zero_action_loss:.6f}")
    print(f"  Ratio (zero_action/normal): {zero_action_loss/normal_loss:.2f}x")
    
    if zero_action_loss / normal_loss < 2.0:
        print("  ⚠️  WARNING: CfC still predicts well without actions!")
        print("     → Likely ignoring action information")
    else:
        print("  ✓ Good: CfC struggles without actions")
        print("     → Actually using action information")
    
    # Test 4: Random weights (baseline)
    print("\n[Test 4] Random weights baseline")
    from module import CfCPredictorV2
    random_predictor = CfCPredictorV2(
        input_dim=32,
        output_dim=32,
        hidden_dim=64,
        num_frames=6,
    ).to(device)
    
    with torch.no_grad():
        pred_emb = random_predictor(all_pixel_emb, all_action_emb)
        pred_target = pred_emb[:, 3:]
        target_emb = all_pixel_emb[:, 3:]
        
        random_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {random_loss:.6f}")
    print(f"  Ratio (random/normal): {random_loss/normal_loss:.2f}x")
    
    if random_loss / normal_loss < 10.0:
        print("  ⚠️  WARNING: Trained model not much better than random!")
        print("     → Model may not have learned meaningful patterns")
    else:
        print("  ✓ Good: Trained model much better than random")
        print("     → Model learned meaningful patterns")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nNormal prediction:     {normal_loss:.6f}")
    print(f"Shuffled input:        {shuffled_loss:.6f} ({shuffled_loss/normal_loss:.2f}x)")
    print(f"Zero actions:          {zero_action_loss:.6f} ({zero_action_loss/normal_loss:.2f}x)")
    print(f"Random weights:        {random_loss:.6f} ({random_loss/normal_loss:.2f}x)")
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    issues = []
    if shuffled_loss / normal_loss < 2.0:
        issues.append("Copying input (shuffle test failed)")
    if zero_action_loss / normal_loss < 2.0:
        issues.append("Ignoring actions (zero-action test failed)")
    if random_loss / normal_loss < 10.0:
        issues.append("Not much better than random")
    
    if issues:
        print("\n⚠️  CfC model has issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendation: Do NOT trust this model for real robot")
        print("                Need to investigate further or retrain")
    else:
        print("\n✓ CfC model passed all tests")
        print("  Safe to use for real robot")


if __name__ == "__main__":
    test_cfc_investigation()
