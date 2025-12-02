"""
Test that JSON serialization works correctly
"""
import json
from predict import AccentPredictor

checkpoint_path = "experiments/attention_cnn_20251201_194410/best_model.pth"
audio_path = "real_data/pe_povo_vai_comer_abbora_melancia_panna_lula_meme.wav"

print("Testing JSON serialization fix...")
print("-" * 70)

predictor = AccentPredictor(checkpoint_path)
result = predictor.predict(audio_path, return_probs=True)

# Try to serialize to JSON
try:
    json_str = json.dumps(result, indent=2)
    print("✅ JSON serialization SUCCESSFUL!")
    print("\nSample of result:")
    print(json_str[:500] + "...")
    
    # Test saving to file
    with open('test_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("\n✅ File save SUCCESSFUL! (test_result.json)")
    
except TypeError as e:
    print(f"❌ JSON serialization FAILED: {e}")

print("-" * 70)
print("Test complete!")

