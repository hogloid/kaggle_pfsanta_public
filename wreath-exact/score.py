"""Evaluation metric for Santa 2023."""
import sys
from pathlib import Path
import pandas as pd
from ast import literal_eval
import fire
from dataclasses import dataclass
from sympy.combinatorics import Permutation
from typing import Dict, List


@dataclass
class Puzzle:
    """A permutation puzzle."""

    puzzle_id: str
    allowed_moves: Dict[str, List[int]]
    solution_state: List[str]
    initial_state: List[str]
    num_wildcards: int


class ParticipantVisibleError(Exception):
    pass


def score_puzzle(puzzle_id, puzzle, sub_solution):
    """Score the solution to a permutation puzzle."""
    # Apply submitted sequence of moves to the initial state, from left to right
    moves = sub_solution.split(".")
    state = puzzle.initial_state
    for m in moves:
        power = 1
        if m[0] == "-":
            m = m[1:]
            power = -1
        try:
            p = puzzle.allowed_moves[m]
        except KeyError:
            raise ParticipantVisibleError(
                f"{m} is not an allowed move for {puzzle_id}."
            )
        state = (p**power)(state)

    # Check that submitted moves solve puzzle
    num_wrong_facelets = sum(not (s == t) for s, t in zip(puzzle.solution_state, state))
    if num_wrong_facelets > puzzle.num_wildcards:
        raise ParticipantVisibleError(f"Submitted moves do not solve {puzzle_id}.")

    # The score for this instance is the total number of moves needed to solve the puzzle
    return len(moves)


PUZZLE_INFO_PATH = Path("../data/puzzle_info.csv")
PUZZLE_PATH = Path("../data/puzzles.csv")

def main(problem_id: int): 
    sub = sys.stdin.read().strip()


    puzzle_info = pd.read_csv(PUZZLE_INFO_PATH, index_col="puzzle_type")
    puzzles = pd.read_csv(PUZZLE_PATH, index_col="id")
    puzzle = puzzles.loc[problem_id]
    
    allowed_moves = literal_eval(
        puzzle_info.loc[puzzle["puzzle_type"], "allowed_moves"]
    )
    allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}

    puzzle = Puzzle(
        puzzle_id=problem_id,
        allowed_moves=allowed_moves,
        solution_state=puzzle["solution_state"].split(";"),
        initial_state=puzzle["initial_state"].split(";"),
        num_wildcards=puzzle["num_wildcards"],
    )

    # Score submission row
    score = score_puzzle(problem_id, puzzle, sub)
    print(f"{sub=}, {score=}")

if __name__ == "__main__":
    fire.Fire(main)
