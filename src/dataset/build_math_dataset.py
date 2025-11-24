import os
import json
import random
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from common import PuzzleDatasetMetadata

cli = ArgParser()

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# Vocabulary: Digits + Operators + Equals + Pad
# PAD is index 0.
CHARSET = "0123456789+-*%()="
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARSET)}
ID2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

class DataProcessConfig(BaseModel):
    output_dir: str = "data/math-arithmetic-medium"
    
    # Dataset Size
    train_size: int = 100000
    test_size: int = 10000
    
    # Sequence Parameters
    seq_len: int = 32        # Fixed length for inputs/outputs
    max_depth: int = 3       # Depth of the expression tree
    max_int: int = 20        # Max value for leaf integers
    
    seed: int = 42

# -----------------------------------------------------------------------------
# Math Generation Logic
# -----------------------------------------------------------------------------

def generate_expression(depth, max_depth, max_int, rng):
    """
    Recursively generates a math expression string and its value.
    """
    # Base case: Return an integer
    if depth >= max_depth or rng.random() < 0.1:
        val = rng.randint(1, max_int)
        return str(val), val

    # Recursive step: Choose an operator
    op = rng.choice(['+', '-', '*', '%'])
    
    left_str, left_val = generate_expression(depth + 1, max_depth, max_int, rng)
    right_str, right_val = generate_expression(depth + 1, max_depth, max_int, rng)

    # Avoid division by zero or modulo zero scenarios if we were doing div
    if op == '%' and right_val == 0:
        right_val = 1
        right_str = "1"

    # Construct string
    expr_str = f"({left_str}{op}{right_str})"
    
    # Calculate value
    if op == '+':
        val = left_val + right_val
    elif op == '-':
        val = left_val - right_val
    elif op == '*':
        val = left_val * right_val
    elif op == '%':
        val = left_val % right_val

    return expr_str, val

def generate_sample(config: DataProcessConfig, rng: random.Random):
    """
    Generates a single (input, label) pair.
    Input:  ((12+4)*3)%7=
    Label:  6
    """
    while True:
        expr_str, val = generate_expression(0, config.max_depth, config.max_int, rng)
        
        # Format Input: "Expression="
        input_str = expr_str + "="
        
        # Format Output: "Value"
        label_str = str(val)
        
        # Filter: Ensure it fits in seq_len
        if len(input_str) <= config.seq_len and len(label_str) <= config.seq_len:
            return input_str, label_str

# -----------------------------------------------------------------------------
# Data Conversion
# -----------------------------------------------------------------------------

def tokenize(text: str, length: int) -> np.ndarray:
    """Converts string to padded numpy array of IDs."""
    ids = [CHAR2ID[c] for c in text]
    arr = np.zeros(length, dtype=np.uint8) # Pad with 0
    arr[:len(ids)] = ids
    return arr

def convert_subset(set_name: str, num_samples: int, config: DataProcessConfig):
    print(f"Generating {set_name} set with {num_samples} samples...")
    
    # Seeding per subset to ensure reproducibility
    rng = random.Random(config.seed + (1 if set_name == 'test' else 0))

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_indices": [0],
        "group_indices": [0],
        "puzzle_identifiers": []
    }

    puzzle_id = 0
    example_id = 0

    for _ in tqdm(range(num_samples)):
        # Generate Math: "((12+4)*3)=" and val 48
        expr_part, val = generate_expression(0, config.max_depth, config.max_int, rng)
        
        # FULL STRING: "((12+4)*3)=48"
        full_str = f"{expr_part}={val}"
        
        if len(full_str) > config.seq_len:
            continue

        # INPUT STRING: "((12+4)*3)=__" (Mask the answer)
        # We keep the "=" sign in the input to prompt the answer
        input_str = f"{expr_part}=" 
        
        # Tokenize
        # 1. Create the full label first (The "Solution")
        lbl_arr = tokenize(full_str, config.seq_len)
        
        # 2. Create the input (The "Problem")
        # We copy the label...
        inp_arr = lbl_arr.copy()
        
        # ...and mask the answer part with 0 (PAD) or a specific MASK token
        # Calculate length of the expression + 1 for '='
        cutoff = len(input_str)
        inp_arr[cutoff:] = 0  # Mask everything after '='

        results["inputs"].append(inp_arr)
        results["labels"].append(lbl_arr) # Label is the FULL sequence

    # Convert to Numpy
    results["inputs"] = np.stack(results["inputs"])
    results["labels"] = np.stack(results["labels"])
    results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)
    results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=len(CHARSET) + 1, # +1 for PAD
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save Data & Metadata
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save Identifiers (Dummy for visualization)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
        
    # Save Vocabulary for reference
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(CHAR2ID, f)

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_subset("train", config.train_size, config)
    convert_subset("test", config.test_size, config)

if __name__ == "__main__":
    cli()