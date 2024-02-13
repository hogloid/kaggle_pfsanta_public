# cube-globe

A solver for `cube` and `globe` puzzles in [Santa 2023 - The Polytope Permutation Puzzle](https://www.kaggle.com/competitions/santa-2023/data) competition.

## Usage

```
cargo run --release -- --id <puzzle_id> --puzzle-info-path <path> --puzzles-path <path>
```

- This solver is written in Rust. You can install Rust from [the official site](https://www.rust-lang.org/tools/install)
- You should provide the path to `puzzle_info.csv` and `puzzle.csv` released in the competition site.
- The first run of this solver on `cube` puzzles should be on odd-sized cubes (this is necessary for creating the cache for "combos")
- The solution will be output to the standard output as in the competition format like `r3.-f1.d0.r2.-d0.f1.d0.-r2.-d2.-d1.f2.r1.-d0.-r1.d0.f2.-d0.f3.d0.-f2.-f0.-d2.f0.-d0.-f0.d2.f0.-r0.f3.-d0.f0.d2.-d3.r1.-d2.d3.r2.-f1.-d3.r0.f1.-r1.-f1.r3.-f0.-r3.f2.r3.f0.-r3.-r0.f0.r0.-f2.-r0.-f0.d3.f1`.
  - For `globe` puzzles, the solution is complete if `mismatch=0` is output to the standard error.
  - For `cube` puzzles, the solution is complete **except for** corner pieces if `mismatch=24`. We need another solver to fix the color of corner pieces.