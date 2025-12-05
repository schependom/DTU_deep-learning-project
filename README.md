# Tiny Recursive Models (TRMs)

| Name                      | Student number |
| ------------------------- | -------------- |
| Vincent Van Schependom    | s251739        |
| Malthe Bresler            | s214631        |
| Jacob Corkill Nielsen     | s204093        |
| Sara Maria Bjørn Andersen | s202186        |

## Introduction

This is the repository of our Deep Learning group project, about the [Tiny Recursive Model](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (TRM), proposed by Alexia Jolicoeur-
Martineau.

The report can be found [here](Report/report.pdf).

## Terminology

| Symbol                 | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| $x$                    | input problem                                                                          |
| $z = z_L$              | latent space variable                                                                  |
| $y = z_H$              | current solution                                                                       |
| $f_\theta = f_L = f_H$ | TRM function (lower-level and higher-level are the same function with same parameters) |
| $n$                    | number of $z \gets f_\theta(x + y +z)$ operations per _full recursion step_            |
| $T-1$                  | number of _full recursion steps_ WITH gradients                                        |
| $N_\text{sup}$         | number of deep supervision steps                                                       |

## Model Architecture

For maximum $N_\text{sup}$ deep supervision steps:

-   For $T-1$ full recursion steps **without gradients**:
    -   For $n$ times:
        -   Update $z \gets f_\theta(x + y + z)$
    -   Update $y \gets f_\theta(y + z)$
-   Final full recursion step **with gradients**:
    -   For $n$ times:
        -   Update $z \gets f_\theta(x + y + z)$
    -   Update $y \gets f_\theta(y + z)$
-   If $\hat{q}_\text{halt}$ indicates to halt, break

\
During training, $\hat{q}_{\text{halt}}$ is calculated with binary cross-entropy loss: predicted correct vs actual correct.
During testing, $\hat{q}_{\text{halt}}$ is ignored and the model is run for $N_\text{sup}$ steps.

## Setup on DTU HPC

> _NOTE_: DTU uses LSF cluster management software. If you are using a different HPC system, the job submission commands will differ.

### Setup SSH

First, create an SSH key pair if you don't have one already:

```bash
ssh-keygen -t ed25519
```

Choose a password for the key when prompted. When the key is created, print the public key with:

```bash
cat ~/.ssh/id_ed25519.pub
```

Now, copy the public key to the HPC server. You can do this by logging into the HPC server using your credentials and adding the public key to the `~/.ssh/authorized_keys` file, or simply using `ssh-copy-id`:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <your-username>@<host>
```

Next, setup the SSH details of your `<host>` in `~/.ssh/config` (Unix), so that you don't have to type in the full details every time you want to connect:

```txt
Host <host>
  HostName <ip-address-or-domain>
  Port <port>
  User <your-username>
  IdentityFile id_ed25519
  AddKeysToAgent yes            # macOS
  UseKeychain yes               # macOS
```

### Connect to HPC

Connect to the HPC using the saved SSH configuration:

```bash
ssh <host>
```

Type in the SSH (and host) password(s) when prompted.

### Move to project folder

Move to Jacob's project folder on the HPC:

```bash
cd /dtu/blackhole/08/156072
```

Then, cd to the project folder:

```bash
cd DTU_deep-learning-project
```

### Python environment using `venv`

First, load the correct Python module:

```bash
module avail python3/3.10 # check available versions
module load python3/3.10.12 # load the right (available) version
```

Also load CUDA (otherwise Adam-atan2 won't install):

```bash
module load cuda/12.6
```

#### Creating a new virtual environment

If the virtual environment does not exist yet, create it (**inside** the project folder!) with:

```bash
python3 -m venv .venv # inside DTU_deep-learning-project/
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Set CUDA_HOME variable such that the devil itself (the adam-atan2 package) installs properly.

```bash
export CUDA_HOME=${CUDA_ROOT:-$(dirname $(dirname $(which nvcc)))}
```

Now install all packages, if not already done:

```bash
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
python3 -m pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
```

#### Activating an existing virtual environment

I already created a virtual environment called `.venv` in the project folder.
Activate it (from within the `DTU_deep-learning-project/` folder!) with:

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the beginning of the command line:

```txt
gbarlogin1(s251739) $ source .venv/bin/activate
(.venv) /dtu/blackhole/08/156072/DTU_deep-learning-project
```

You can (but right now don't have to) deactivate the environment with:

```bash
deactivate
```

## Technical Implementation Details

### Data Hierarchy

A critical component of training on many _variants_ of puzzles is the data hierarchy. The datasets are organized into three levels, preventing _data leakage_.

The `PuzzleDataset` class implements this hierarchy:

1.  **Groups**

    -   Collections of puzzles that are related
    -   Atomic unit for train/test splitting
    -   If Group $x$ is in the training set, no puzzles from Group $x$ appear in the test set
    -   Prevent data leakage
    -   e.g., augmentations of the same base puzzle

2.  **Puzzles**

    -   "Variants" of the same problem
    -   Specific augmentation or transformation of the Group
    -   Increase dataset size and forces model to learn invariant rules, regardless of representation

3.  **Examples**

    -   Input-Output pairs
    -   In sudoku and maze, **1 example per puzzle**

**Example:**

```txt
Group 0:
  ├─ Puzzle 0: [Example 0, Example 1, Example 2]
  ├─ Puzzle 1: [Example 3, Example 4]
  └─ Puzzle 2: [Example 5]
Group 1:
  ├─ Puzzle 3: [Example 6, Example 7, Example 8, Example 9]
  └─ Puzzle 4: [Example 10]
```

### File Structure

When you generate a dataset, you get these files:

```txt
data/(puzzle)/
├── train/
│   ├── dataset.json                    # Metadata
│   ├── all__inputs.npy                 # Input sequences [num_examples, seq_len]
│   ├── all__labels.npy                 # Target sequences [num_examples, seq_len]
│   ├── all__puzzle_identifiers.npy     # Which puzzle each belongs to [num_puzzles]
│   ├── all__puzzle_indices.npy         # Start index of each puzzle [num_puzzles+1]
│   └── all__group_indices.npy          # Start index of each group [num_groups+1]
└── test/
    └── (same structure)
```

`dataset.json`:

-   Contains the "schema" of the data.
-   `vocab_size`: Total number of distinct token IDs (e.g., 11 for Sudoku: 0=pad, 1-9=digits).
-   `seq_len`: The fixed length of the input/output vector (e.g., 81 for Sudoku, 900 for Maze).

`all__inputs.npy`:

-   Shape `[N_samples, seq_len]`.
-   Contains the initial problem state.
-   For Sudoku: The board with clues (1-9) and blanks (0).
-   For Maze: The walls (#), start (S), and goal (G).

`all__labels.npy`:

-   Shape `[N_samples, seq_len]`.
-   Contains the target solution state (**final** solution!).
-   For Sudoku: The completed board (1-9).
-   For Maze: The path from start to goal (., S, G).
-   The model does **not** predict one step.
    -   It takes the `input` (Unsolved),
    -   processes it internally for K cycles,
    -   the output head is trained to match `label` (Fully Solved).

`all__puzzle_indices`, `all__group_indices`:

-   Used for data augmentation.
-   If you generate 8 augmentations (rotations/flips) of one Sudoku board, they share a `group_index`.
-   This ensures that when splitting Train/Test, you don't put a rotated version of a training puzzle into the test set (data leakage).

How the indices work:

-   E.g. `puzzle_indices = [0, 3, 5, 6, 10]` means:
    -   Puzzle 0 has examples from index 0 to 2
    -   Puzzle 1 has examples from index 3 to 4
    -   Puzzle 2 has examples from index 5 to 5
    -   Puzzle 3 has examples from index 6 to 9
-   E.g. `group_indices = [0, 3, 6]` means:
    -   Group 0 has puzzles from index 0 to 2
    -   Group 1 has puzzles from index 3 to 5

### Training

In training Mode, `test_set_mode=False`.

The `_iter_train` function does this:

1. **Shuffle groups** randomly each epoch
2. For each group, **randomly pick one puzzle** from that group
3. From that puzzle, **randomly sample examples** without replacement
4. Pack examples into batches of size `global_batch_size`
5. Distribute across GPUs (each GPU gets `local_batch_size` examples)

### Testing

In testing Mode, `test_set_mode=True`.

The `_iter_test` function does this:

1. Go through **all examples sequentially** (no shuffling)
2. Pack into batches
3. Distribute across GPUs

This ensures we evaluate on every single example exactly once.

## Experiments

We investigated how well we could recreate the results of the paper. We were able to obtain a Sudoku accuracy of $84.3\%$, just shy of the papers $87.4\%$, most likely due to 24h train limit.

We then continued to investigate performance for a CSP, namely $N$-Queens, as well as other types of puzzles, like the Towers of Hanoi, Math Arithmetic and Chess Move Prediction to test the limits of the TRM.

All jobscripts can be found in this main folder as `job-[experiment].sh`.

### $N$-Queens

The $N$-Queens problem involves placing N queens on an N×N chessboard such that no two queens threaten each other. The TRM functions as a **pattern completion engine**, not a next move predictor.

#### The choice of $N=12$

For $N=12$, there are 14,200 unique valid solutions. This size is computationally manageable while still being complex enough to challenge the model.

#### Input

-   A $12 \times 12$ grid (flattened to length 144).
-   Some cells contain a `Queen` (Value 2).
-   All other cells are marked as `Unknown`/`Empty` (Value 0 or 1).
-   This represents a "partial thought" or a set of constraints: "I know a `Queen` goes here and here, but I don't know the rest."

#### Prediction

-   The model outputs the entire $12 \times 12$ grid simultaneously.
-   Every cell is filled with a definite decision: `Queen` or `Empty`.
-   The goal is for this final grid to satisfy the global N-Queens constraints (no attacks on rows, columns, or diagonals) while respecting the initial "hints" provided in the input.

#### Dataset Structure

The dataset uses the standard TRM 3-level hierarchy, but adapted for the N-Queens generation process:

-   **The base solution:**

    -   Before creating groups, we solve the N-Queens problem to find all unique Base Solutions (e.g., for $N=12$, there are 14,200 unique completed boards).

-   **Level 1: Group (specific problem instance)**

    -   A single **masked problem** derived from a base solution.

    -   Example: "Solution #42 with 50% of queens hidden."

    -   In the training loop, we generate multiple different mask patterns (different difficulty levels) for the same base solution. Each mask pattern becomes its own group.

-   **Level 2: Puzzle (augmentations)**

    -   Inside each group, we generate multiple puzzles.

    -   These correspond to the Dihedral Symmetries (Rotations 0/90/180/270 + Flips) of that specific masked problem.

    -   Because the $N$-Queens constraint is symmetric (a rotated solution is still a valid solution), these puzzles are mathematically valid.

-   **Level 3: Example**

    -   The actual input/label tensors.

Thus, the overall structure looks like this:

```txt
Split: Train Set (Unique Base Solutions A..Y)
   |
   ├── Group 1 (Solution A, Mask Pattern 1)
   |     ├── Puzzle 1 (0° Rotation)
   |     ├── Puzzle 2 (90° Rotation)
   |     └── ... (8 variants)
   |
   ├── Group 2 (Solution A, Mask Pattern 2)
   |     └── ... (8 variants)
   |
   └── Group 3 (Solution B, Mask Pattern 1) ...

Split: Test Set (Unique Base Solution Z)
   |
   └── Group 100 (Solution Z, Mask Pattern 1)
         └── Puzzle 1 (0° Rotation only)
```

Note that we limit the test set to the **"canonical" orientation only** (no transformations and a single mask pattern), to reduce evaluation time. Generating 8x augmentations for evaluation is unnecessary. If the model has learned the concept of $N$-Queens from the training set (which included rotations), it should be able to solve the 0-degree test case perfectly.

#### Running the job

```bash
bsub < job-n_queens.sh          # All solutions
bsub < job-n_queens_unamb.sh    # Unambiguous solutions only
```

### Sudoku Extreme

To run Sudoku Extreme, we first generated the dataset as stated in the authors [`src/README.md`](src/README.md). After this, we simply ran the jobscript `job-sudoku.sh` on DTU HPC to obtain the results discussed in the project.

We also tried to recreate the results for the Maze dataset. The dataset could again be prepared by following the code as specified in the original [`src/README.md`](src/README.md). Our last test job script can be seen in the file `maze.sh`. However, this required too many GPUs and thus never ran. We also tried only requesting one GPU, but when trying this we ran out of GPU memory.

### Hanoi (didn't work very well)

#### Generate the data

From the `src/` folder, run:

```bash
# Generate and visualise datasets for both encodings
./build_hanoi.sh

# Or generate individually
python dataset/build_hanoi_dataset.py --encoding action --out data/hanoi_action
python dataset/build_hanoi_dataset.py --encoding state --out data/hanoi_state

# Custom disk ranges
python build_hanoi_dataset.py --encoding action --train-min 2 --train-max 5 --test-min 6 --test-max 8
```

View the usage information with:

```bash
python dataset/build_hanoi_dataset.py --help
```

#### Visualize the data

The `visualize_hanoi.py` script will _automatically_ detect which encoding was used (action or state) and visualize accordingly.

Assuming you ran `./build_hanoi` previously, from the `src/` folder, run:

```bash
python dataset/visualize_hanoi.py --dir data/hanoi_action/train
python dataset/visualize_hanoi.py --dir data/hanoi_state/train
```

### Chess Move Prediction

The 200k subset of the lichess open database: https://database.lichess.org/#evals exists in `chess_200k.csv`, but if a different size subset is needed, it can be created via `create_chess_csv.py`.

The chess dataset can be built in the right format (from the `src/` folder) via

```bash
./build_chess.sh
```

Then run the job script

```bash
bsub < job-chess.sh
```

### Math Arithmetic

#### Generate the data

From `src/`, run:

```bash
./build_arithmetic.sh
```

#### Run the job

From the main repo folder, run:

```bash
bsub < job-arithmetic.sh
```
