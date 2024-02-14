

### Wreath
For puzzles smaller than or equal to 21/21, the optimal solution (with no wildcards) can be obtained through a bidirectional search. This entails performing a breadth-first search from both the initial state and the solution state; once they meet at a common state, the sequence of actions from the initial state is connected with the reversed sequence of actions from the solution state. Please note that this process can consume considerable memory, hence the encoding of the state into 128-bit integers is advised. You may refer corresponding source code in `wreath-exact`

For puzzles with 8 wildcards, we have devised a specific algorithm. Let's focus on the 100x100 puzzle (the explanation still applies, but it's simpler for a 33x33 puzzle). You may google ‘hungarian puzzle’ to see what the puzzle looks like.

First of all, let's examine the solution state. The left ring consists of 'A' cells, while the right ring consists of 'B' cells. There are 'C' cells at the two intersection points. The distance (total number of cells + 1) between two intersections is 25 in the left ring and 26 in the right ring. Let’s say one of the intersection cells is ‘bottom’, and the other is ‘top’. To simplify, we can recolor 'C' cells as 'B', which will cost at most 4 mismatch errors.

Let’s enumerate the indices of the 'B' cells from 0 to 99 in the clockwise order so that the 0-th cell is in the bottom intersection point, and the 26-th cell is in the top intersection point at first.
A cell in the right ring can be recolored as 'B' by relocating the cell to the bottom intersection point, revolving the left ring, and then returning the cell back to its original position. However, this can risk changing the state of the right cell, which is in the top intersection point.

That said, it's possible to rectify the state of cells in the right ring in the following order of indices: 0, 26, 52, 76, 4, 30, …, 48, 74. After this sequence, only the 0th cell is wrong, because rectifying 26*k (mod 100)-th cell risks only 26*(k+1)-th cell, by putting 26*k-th, and 26*(k+1)-th cell in the intersection point when revolving the left ring.

Apply the same rule for 1, 27, 53, 77, 5, 31 …, 49, 75. This process may induce at most 2 mismatch errors for even/odd cells, yielding a total of 8 mismatch errors at the most.

Following this algorithm as is results in too many actions, considering it necessitates 26 * 100 actions for the right ring. To alleviate this, an additional preprocessing step is carried out to correct randomly chosen sets of cells in the right ring in ascending order from 0 to 99. That is, do the following:
toCorrect := random array of [0,1] of length 100
for i in range(100):
	if toCorrect[i]:
		revolve the left ring so that the bottom intersection point becomes ‘B’
	revolve the right ring once in counterclockwise

This approach requires at most 100 steps for the right ring, and aids in reducing the number of actions in subsequent processes, because for example, after rectifying the state of 26-th cell, if the state of 52, 76, 4-th cell is already ‘B’, only 4 steps to revolve the right ring is required to get the 30-th cell to the bottom intersection point. Trying toCorrect sequence about 10**6 times gives us a solution with length 681.
