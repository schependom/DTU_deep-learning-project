import os
import json
import numpy as np

def verify_dataset(base_path):
    print(f"VERIFYING DATASET AT: {base_path}")
    print("=" * 60)

    if not os.path.exists(base_path):
        print(f"❌ ERROR: Base path '{base_path}' does not exist.")
        return

    subsets = ["train", "test"]
    
    for subset in subsets:
        subset_path = os.path.join(base_path, subset)
        print(f"\nChecking subset: [{subset}]")
        print("-" * 30)

        if not os.path.exists(subset_path):
            print(f"❌ Missing directory: {subset_path}")
            continue

        # 1. Check Metadata
        json_path = os.path.join(subset_path, "dataset.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                print(f"✅ dataset.json found.")
                print(f"   - Sequence Length: {meta.get('seq_len')}")
                print(f"   - Vocab Size:      {meta.get('vocab_size')}")
                print(f"   - Total Groups:    {meta.get('total_groups')}")
            except Exception as e:
                print(f"❌ Error reading dataset.json: {e}")
        else:
            print(f"❌ Missing file: dataset.json")

        # 2. Check Data Files
        files_to_check = [
            "all__inputs.npy",
            "all__labels.npy", 
            "all__group_indices.npy"
        ]

        for fname in files_to_check:
            fpath = os.path.join(subset_path, fname)
            if os.path.exists(fpath):
                try:
                    data = np.load(fpath)
                    print(f"✅ {fname:<25} | Shape: {str(data.shape):<15} | Dtype: {data.dtype}")
                    
                    if fname == "all__inputs.npy":
                        unique_tokens = np.unique(data)
                        print(f"   - Unique Input Tokens: {unique_tokens} (Expect: [0, 2, 3] or similar)")
                    
                    if fname == "all__group_indices.npy":
                         num_groups = len(np.unique(data))
                         print(f"   - Unique Groups Found: {num_groups}")

                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")
            else:
                print(f"❌ Missing file: {fname}")

    print("\n" + "=" * 60)
    print("Verification Complete.")

if __name__ == "__main__":
    # Adjust N if you used a different size
    N_SIZE = 12 
    DATA_DIR = f"data/n_queens_unamb_{N_SIZE}"
    verify_dataset(DATA_DIR)