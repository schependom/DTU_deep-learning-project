import csv
from datasets import load_dataset

def extract_best_move(line_str):
    if not line_str:
        return None
    # Split the PV line by spaces and grab the first move
    return line_str.split()[0]

def main():
    # 1. Load in streaming mode (no massive download required)
    print("Connecting to dataset stream...")
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    # 2. Limit to the first 100,000 examples
    # The .take() method ensures the script stops processing automatically
    subset = ds.take(200_000)

    output_file = "chess_200k.csv"
    
    print(f"Extracting first 100k rows to {output_file}...")
    
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
                print(f"Processed {count} rows...", end="\r")

    print(f"\nSuccessfully saved {count} rows to {output_file}.")

if __name__ == "__main__":
    main()