run_name="pretrain_cpu_small_sudoku"
export DISABLE_COMPILE=1
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-small]" \
global_batch_size=16 \
evaluators="[]" \
epochs=10 eval_interval=5 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True