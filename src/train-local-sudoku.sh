export DISABLE_COMPILE=1
export PYTHONWARNINGS="ignore"
run_name="pretrain_mlp_FAST_TEST"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-small]" \
evaluators="[]" \
epochs=10 eval_interval=10 \
global_batch_size=32 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=1 \
arch.H_cycles=1 \
arch.L_cycles=2 \
+run_name=${run_name} ema=True