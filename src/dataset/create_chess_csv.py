import csv
from datasets import load_dataset

"""
This document loads a subsection of the lichess chess move set and writes it to a csv, which is then used for build_chess_dataset.py
"""

N = 200_000

def extract_best_move(line_str):
    if not line_str:
        return None
    return line_str.split()[0]

def main():
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    subset = ds.take(N)

    output_file = "chess_200k.csv"
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["FEN", "Best_Move"])
        
        count = 0
        for row in subset:
            fen = row.get("fen")
            line = row.get("line")
            
            best_move = extract_best_move(line)
            
            if fen and best_move:
                writer.writerow([fen, best_move])
            
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows...\n")


if __name__ == "__main__":
    main()