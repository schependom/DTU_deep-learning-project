import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

cli = ArgParser()

class VisualizeConfig(BaseModel):
    data_dir: str = "data/math-arithmetic"
    split: str = "test"
    num_samples: int = 10

@cli.command(singleton=True)
def main(config: VisualizeConfig):
    # 1. Load Vocabulary
    vocab_path = os.path.join(config.data_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Error: Vocab file not found at {vocab_path}. Did you run build_math_dataset.py?")
        return
    
    with open(vocab_path, "r") as f:
        char2id = json.load(f)
    # Invert mapping: ID -> Char. 0 is Padding.
    id2char = {v: k for k, v in char2id.items()}
    id2char[0] = "_" # Represent padding as underscore for visibility

    # 2. Load Data Arrays
    subset_path = os.path.join(config.data_dir, config.split)
    inputs_path = os.path.join(subset_path, "all__inputs.npy")
    labels_path = os.path.join(subset_path, "all__labels.npy")

    if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
         print(f"Error: Data files not found in {subset_path}")
         return

    inputs = np.load(inputs_path)
    labels = np.load(labels_path)

    print(f"Successfully loaded {len(inputs)} samples from '{config.split}' set.")
    print(f"Visualizing {config.num_samples} random samples...\n")
    print("-" * 40)
    
    # 3. Pick random samples
    indices = np.random.choice(len(inputs), min(config.num_samples, len(inputs)), replace=False)

    for idx in indices:
        inp_ids = inputs[idx]
        lbl_ids = labels[idx]

        # Decode: Join characters, ignoring padding for readability (or keeping it as '_')
        # Here we strip padding (0) to see the actual math
        inp_str = "".join([id2char.get(i, "?") for i in inp_ids if i != 0])
        lbl_str = "".join([id2char.get(i, "?") for i in lbl_ids if i != 0])
        
        # Raw (padded) view for debugging
        # raw_inp = "".join([id2char.get(i, "?") for i in inp_ids])

        print(f"Sample #{idx}:")
        print(f"  Input:  {inp_str}")
        print(f"  Target: {lbl_str}")
        print("-" * 40)

if __name__ == "__main__":
    cli()