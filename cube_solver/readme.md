# install solvers

## Installing 3x3x3 solver

- See [official repository](https://github.com/dwalton76/rubiks-cube-NxNxN-solver)
- Although venv is recommended, you don't have to use it.
- If `cd rubiks-cube-NxNxN-solver && python rubiks-cube-solver.py` works, it is OK
- When executing `rubiks-cube-solver.py`, workdir must be the path where `rubiks-cube-solver.py` is located. If you save it in a location other than this directory, specify the solver path with `--solver_path`

## Installing 4x4x4 solver
- See [Official Repository](https://github.com/cs0x7f/TPR-4x4x4-Solver)
- You can test the solver by `java -cp .:threephase.jar:twophase.jar solver UUURUUUFUUUFUUUFRRRBRRRBRRRBRRRBRRRDFFFDFFFFFDDDDBDDDBDDDBDDDLFFFFLLLLLLLLLLLLULLLUBBBUBBBBBBB`


# run

## List of special idx issues

[200-204] 4x4x4 with ABABABA
[205-209] 4x4x4 with N1,N2...
[235-239] 5x5x5 with ABABAB
[240-244] 5x5x5 with N1,N2...
[255] 6x6x6 with ABABABA
[256] 6x6x6 with N1,N2...
[282] 33x33x33 ABABAB
[283] 33x33x33 N1,N2...

## All questions other than those listed above
Add skip_alias option
`python pysolver.py --id 281(your_id) --skip_alias --initial_moves=[your_move]`

## For those with N1,N2... and an odd cube size
add surf option
`python pysolver.py --id 283 --surf --initial_moves=[your_move]`

# For odd size ABABAB...
Do not add skip_alias option
`python pysolver.py --id 282 --initial_moves=[your_move]`

# Regarding ABABAB... of even size
Just use dwalton's solver
`python pysolver.py --id 200 --js`
