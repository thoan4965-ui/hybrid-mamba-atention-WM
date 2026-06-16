"""
Investigate AR model - check if it's cheating or actually learning
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load_model


def test_ar_investigation():
    device = "cpu"
    
    # Load AR epoch 10
    print("Loading AR epoch 10...")
    ar_model = load_model("D:/ai_training/MODELS/ar_models/vit_ar_epoch_10.ckpt",
                          model_type="ar", device=device)
    
    encoder = ar_model["encoder"]
    predictor = ar_model["predictor"]
    action_encoder = ar_model["action_encoder"]
    
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
        
        normal_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {normal_loss:.6f}")
    
    # Test 2: Shuffle input frames
    print("\n[Test 2] Shuffle input frames (check if copying)")
    shuffled_pixels = pixels[:, torch.randperm(T)]  # Shuffle time dimension
    shuffled_pixels_flat = shuffled_pixels.reshape(B * T, C, H, W)
    
    with torch.no_grad():
        shuffled_pixel_emb = encoder(shuffled_pixels_flat).reshape(B, T, -1)
        
        history_emb = shuffled_pixel_emb[:, :3]
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
        target_emb = all_pixel_emb[:, 3:]  # Still compare with original target
        
        shuffled_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {shuffled_loss:.6f}")
    print(f"  Ratio (shuffled/normal): {shuffled_loss/normal_loss:.2f}x")
    
    if shuffled_loss / normal_loss < 2.0:
        print("  ⚠️  WARNING: AR still predicts well with shuffled input!")
        print("     → Likely copying input instead of learning dynamics")
    else:
        print("  ✓ Good: AR struggles with shuffled input")
        print("     → Actually learning temporal dynamics")
    
    # Test 3: Zero actions
    print("\n[Test 3] Zero actions (check if using actions)")
    zero_actions = torch.zeros_like(actions)
    zero_action_emb = action_encoder(zero_actions)
    
    with torch.no_grad():
        history_emb = all_pixel_emb[:, :3]
        history_act = zero_action_emb[:, :3]
        predictions = []
        
        for i in range(3):
            pred = predictor(history_emb, history_act)[:, -1:]
            predictions.append(pred)
            history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
            history_act = torch.cat([history_act[:, 1:], zero_action_emb[:, 3 + i:3 + i + 1]], dim=1)
        
        pred_target = torch.cat(predictions, dim=1)
        target_emb = all_pixel_emb[:, 3:]
        
        zero_action_loss = (pred_target - target_emb).pow(2).mean().item()
    
    print(f"  pred_loss: {zero_action_loss:.6f}")
    print(f"  Ratio (zero_action/normal): {zero_action_loss/normal_loss:.2f}x")
    
    if zero_action_loss / normal_loss < 2.0:
        print("  ⚠️  WARNING: AR still predicts well without actions!")
        print("     → Likely ignoring action information")
    else:
        print("  ✓ Good: AR struggles without actions")
        print("     → Actually using action information")
    
    # Test 4: Random weights (baseline)
    print("\n[Test 4] Random weights baseline")
    random_predictor = type(predictor)(
        num_frames=3,
        depth=1,
        heads=2,
        mlp_dim=96,
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        dim_head=16,
    ).to(device)
    
    with torch.no_grad():
        history_emb = all_pixel_emb[:, :3]
        history_act = all_action_emb[:, :3]
        predictions = []
        
        for i in range(3):
            pred = random_predictor(history_emb, history_act)[:, -1:]
            predictions.append(pred)
            history_emb = torch.cat([history_emb[:, 1:], pred], dim=1)
            
            if i < 2:
                next_act = all_action_emb[:, 3 + i:3 + i + 1]
                history_act = torch.cat([history_act[:, 1:], next_act], dim=1)
        
        pred_target = torch.cat(predictions, dim=1)
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
        print("\n⚠️  AR model has issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendation: Do NOT trust this model for real robot")
        print("                Need to investigate further or retrain")
    else:
        print("\n✓ AR model passed all tests")
        print("  Safe to use for real robot")


if __name__ == "__main__":
    test_ar_investigation()
