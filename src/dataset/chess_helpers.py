

# #Chess puzzles dataset: https://huggingface.co/datasets/Lichess/chess-puzzles
# #Chess positions dataset: https://www.kaggle.com/code/gabrielhaselhurst/chess-dataset/input
# #Chess positions AND evaluations: https://huggingface.co/datasets/Lichess/chess-position-evaluations
# #Full chess games: https://github.com/angeluriot/Chess_games

import csv
from tqdm import tqdm
import time
from stockfish import Stockfish





# stockfish.set_fen_position("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
# print(stockfish.get_top_moves(3))

def rows_with_best_move(csv_path: str, stockfish: Stockfish, N_ROWS: int = 10_000, fen_col: str = "FEN", eval_col: str = "Evaluation"):
    rows = []
    row_count = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        # detect columns
        header = reader.fieldnames or []

        for r in tqdm(reader):

            if row_count >= N_ROWS:
                break

            fen = r[fen_col].strip()

            # --- TRY PARSING EVAL AS FLOAT ---
            eval_raw = r.get(eval_col, "").strip()
            try:
                eval_v = float(eval_raw)
            except ValueError:
                # ignore rows that don't contain a valid float
                continue
            
            #================================= Stockfish logic =================================# 
            stockfish.set_fen_position(fen)
            move = stockfish.get_best_move()

            rows.append({
                "FEN": fen,
                "Evaluation": eval_v,
                "Best_Move": move
            })

            row_count+=1
    return rows

def save_csv(save_path: str, content: list[dict]):
    fieldnames = ["FEN", "Evaluation", "Best_Move"]

    try:
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row (column names)
            writer.writeheader()

            # Write all the data rows
            writer.writerows(content)

        print(f"Successfully saved {len(content)} rows to {save_path}")

    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")


if __name__ == "__main__":
    stockfish = Stockfish(path=r"C:\Users\malth\Documents\stockfish\stockfish-windows-x86-64-avx2.exe",
                        depth=20, 
                        parameters={"Threads": 8, 
                                    "Hash": 8192})
    
    dataset_csv = rows_with_best_move(csv_path=r"C:\Users\malth\Documents\DTU\Master\Andet Semester\Deep Learning\DTU_deep-learning-project\src\dataset\chessData.csv", stockfish=stockfish, N_ROWS=10_000)

    save_csv(save_path = "./src/dataset/chess_moves_2.csv", content=dataset_csv)
#TODO:
# - PRECOMPUTING:
# - load game state (s)
# - get evaluation and best move
# - Use best move to get new game state (s+)
# - get evaluation of new game state (s+)
#
#
#
# IN TRAINING LOOP:P
# - get move from TRM
# - use move to get state (s*)
# - get evaluation from (s*)
# - calculate loss as difference in evaluation (take white / black into account)



#Andre sæt:
# - https://huggingface.co/datasets/Lichess/chess-position-evaluations
# - https://huggingface.co/datasets/Lichess/chess-puzzles
