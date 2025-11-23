import torch
import time
import os

# --- CONFIG ---
# Update this path to match your experiment
VECTOR_FILE_PATH = 'experiments/my_experiment/extraction/cognitive_state/confidence/vectors/probe_layer1.pt'
# ---

print("--- Simple Torch Load Test on Failing File ---")
print(f"File path: {os.path.abspath(VECTOR_FILE_PATH)}")

if not os.path.exists(VECTOR_FILE_PATH):
    print(f"\nERROR: File does not exist at the specified path.")
    exit()

print(f"File size: {os.path.getsize(VECTOR_FILE_PATH)} bytes")
print("Attempting to load tensor with torch.load()...")

start_time = time.time()
try:
    tensor = torch.load(VECTOR_FILE_PATH, weights_only=True)
    end_time = time.time()
    print("\n✅ Successfully loaded tensor!")
    print(f"   Time taken: {end_time - start_time:.4f} seconds")
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   Tensor dtype: {tensor.dtype}")

except Exception as e:
    end_time = time.time()
    print(f"\n❌ Failed to load tensor after {end_time - start_time:.2f} seconds.")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {e}")

print("--------------------------------------------")