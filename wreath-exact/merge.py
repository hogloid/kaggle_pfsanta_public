from pathlib import Path
import pandas as pd

FLOWLIGHT_DIRECTORY = Path(__file__).parent
OUTPUT_TXTS_DIR = FLOWLIGHT_DIRECTORY / "outputs"
OUTPUT_CSV_PATH = FLOWLIGHT_DIRECTORY.parent / "submission_candidates" / "flowlight" / "wreath_exact.csv"

def main(): 
    ids = []
    moves = []
    for i in range(284, 336): 
        file = OUTPUT_TXTS_DIR / f"{i}.txt"
        if not file.exists(): 
            continue
        sol_moves = file.read_text().strip()
        if len(sol_moves) > 0: 
            ids.append(i)
            moves.append(sol_moves)
    solution = pd.DataFrame({
        "id": ids, 
        "moves": moves
    })
    solution.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__": 
    main()    