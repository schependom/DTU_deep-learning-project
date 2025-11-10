# Tiny Recursive Models (TRMs)

## Setup (local, CPU friendly)

### Python Environment

Create a new conda environment:

```bash
conda create -n trm python=3.10 -y
```

Activate the environment:

```bash
conda activate trm
```

Install the required packages:

```bash
pip install --no-cache-dir -r requirements-macos.txt
```

## Setup (HPC)

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
  AddKeysToAgent yes
  UseKeychain yes
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
module load python/3.10.2 # load the right (available) version
```

#### Activating the existing virtual environment

I already created a virtual environment called `.venv` in the project folder.
Activate it (from within the `DTU_deep-learning-project/` folder!) with:

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the beginning of the command line.
You can (but right now don't have to) deactivate the environment with:

```bash
deactivate
```

#### Installing (extra) packages if needed

To install packages inside the `venv`, use:

```bash
python3 -m pip install <packages>
```

We are used to simply using `pip3`, but this is the recommended and correct way of installing packages. The `-m` flag in Python allows us to run modules as scripts. This way we ensure that the module is located in your current python environment, not the global python installation.

#### Creating a new virtual environment

If the virtual environment does not exist yet (it should!), create it (**inside** the project folder!) with:

```bash
python3 -m venv .venv # inside DTU_deep-learning-project/
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Now simply install from the `requirements.txt` file:

```bash
python3 -m pip install -r requirements.txt
```

To update the packages using the `requirements.txt` file:

```bash
python3 -m pip install --upgrade -r requirements.txt
```

## Datasets

### Hierarchy

How does the `PuzzleDataset` class work?
The dataset has a 3-level hierarchy:

-   Examples (lowest level): Individual training samples (input-output pairs)
-   Puzzles (middle level): Collections of examples from the same puzzle
-   Groups (highest level): Collections of puzzles that should be sampled together

For example:

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
data/hanoi/
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

How the indices work:

-   E.g. `puzzle_indices = [0, 3, 5, 6, 10]` means:
    -   Puzzle 0 has examples from index 0 to 2 (indices 0, 1, 2 -> 3 examples)
    -   Puzzle 1 has examples from index 3 to 4 (indices 3, 4 -> 2 examples)
    -   Puzzle 2 has examples from index 5 to 5 (index 5 -> 1 example)
    -   Puzzle 3 has examples from index 6 to 9 (indices 6, 7, 8, 9 -> 4 examples)
-   E.g. `group_indices = [0, 3, 6]` means:
    -   Group 0 has puzzles from index 0 to 2 (3 puzzles: 0, 1, 2)
    -   Group 1 has puzzles from index 3 to 5 (3 puzzles: 3, 4, 5)

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

## Small CPU-size datasets

### Small sudoku

To generate a small sudoku dataset for CPU testing, run:

```bash
python dataset/build_easy_sudoku_dataset.py --output-dir data/sudoku-small --subsample-size 100 --num-aug 1
```

To train and evaluate on this dataset, run:

```bash
bash train-local-sudoku.sh
```

## Hanoi

### Generate the data

```bash
cd src
python dataset/build_hanoi_dataset.py <options>
```

You can view the options by running:

```bash
python dataset/build_hanoi_dataset.py --help
```

This will generate the `train/` and `test/` data in the folder `src/data/hanoi/`.

### Training

To train on one GPU, from the `src/` folder, run:

```bash
run_name="pretrain_mlp_t_hanoi"
python pretrain.py \
arch=trm \
data_paths="[data/hanoi]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```
