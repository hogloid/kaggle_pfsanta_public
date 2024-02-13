from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
import numpy as np
import os
import subprocess
import re

from itertools import permutations
import twophase.solver as sv

input_str = 'ABCDEF'
abcdef_list = list(permutations(input_str))
ulist = 'UFRBLD'

"""
cubeメモ:

面の並び方は以下のようになってる

# 0
#4123
# 5

各面については

         0   1  2
         3   4  5
         6   7  8
36 37 38 9  10 11 18 19 20 27 28 29
39 40 41 12 13 14 21 22 23 30 31 32
42 43 44 15 16 17 24 25 26 33 34 35
         45 46 47
         48 49 50
         51 52 53

# f0, r0, d0は125を時計回しすることに相当
"""


### 汎用関数 ###

def get_moves(puzzle_type: str, puzzle_info_path:str="data/puzzle_info.csv") -> dict[str, tuple[int, ...]]:
    """
    puzzle_infoの情報に則って合法手のリストを取得する
    """
    moves = eval(pd.read_csv(puzzle_info_path).set_index("puzzle_type").loc[puzzle_type, "allowed_moves"])
    for key in list(moves.keys()):
        moves["-" + key] = list(np.argsort(moves[key]))
    return moves


def get_rev(act:str) -> str:
    """
    アクションを逆転させる
    """
    if isinstance(act, list):
        out = []
        for i in range(len(act)):
            out.append(get_rev(act[len(act)-1-i]))
        return out

    if act.startswith("-"):
        return act[1:]
    else:
        return "-" + act
    
def solve_pll(cube_size) -> List[str]:
    assert cube_size % 2 == 0
    moves = []
    for i in range(cube_size // 2 - 1):
        moves.extend([f"r{i+1}", f"r{i+1}"])
    moves.extend(["d0", "d0"])
    for i in range(cube_size // 2 - 1):
        moves.extend([f"r{i+1}", f"r{i+1}"])
    
    for i in range(cube_size // 2):
        moves.extend([f"d{i}", f"d{i}"])

    for i in range(cube_size // 2 - 1):
        moves.extend([f"r{i+1}", f"r{i+1}"])
    
    for i in range(cube_size // 2 - 1):
        moves.extend([f"d{i+1}", f"d{i+1}"])
    return moves

def solve_oll(cube_size) -> List[str]:
    assert cube_size % 2 == 0
    moves = []
    
    for i in range(cube_size // 2):
        moves.extend([f"r{i}"])

    moves.extend(["d*", "d*"])

    for i in range(cube_size // 2):
        moves.extend([f"r{i}"])

    moves.extend(["f0", "f0"])


    for i in range(cube_size // 2):
        moves.extend([f"r{i}"])

    moves.extend(["f0", "f0"])


    for i in range(cube_size // 2):
        moves.extend([f"-r{i}"])

    moves.extend(["f0", "f0"])



    for i in range(cube_size // 2):
        moves.extend([f"-r{cube_size-1-i}"])

    moves.extend(["f0", "f0"])


    for i in range(cube_size // 2):
        moves.extend([f"-r{i}"])

    moves.extend(["f0", "f0"])


    for i in range(cube_size // 2):
        moves.extend([f"r{i}"])

    moves.extend(["f0", "f0"])

    for i in range(cube_size // 2):
        moves.extend([f"-r{i}"])

    moves.extend(["f0", "f0"])

    for i in range(cube_size // 2):
        moves.extend([f"-r{i}"])

    return moves


def solve_by_solver(state:Union[str, List[str]], solver_path:str="/home/shiku/AI/kaggle/santa2023/rubiks-cube-NxNxN-solver", solver_path_444="/home/shiku/AI/kaggle/santa2023/TPR-4x4x4-Solver",force_odd=False, force3=False) -> List[str]:
    """
    solverを使って問題を解く。stateはA-Fの文字列で構成されている必要がある(e.g. AAABBCD...)
    solver_pathにはsolverのパスを置く。solverの導入方法は
    https://www.kaggle.com/code/seanbearden/solve-all-nxnxn-cubes-w-traditional-solution-state
    
    とかを参照。与えたstateから全ての面を揃えた状態になるために必要なsanta準拠のコマンドをlist[str]形式で出力する
    """
    pattern = re.compile(r'^(\d*)?([A-Za-z])(w)?([\'2])?$')

    for sid, perm in enumerate(abcdef_list, force3):
        U_dict = {perm[i] : ulist[i] for i in range(6)}

        def state2ubl(state, even, force3=False):
            state_split = state
            dim = int(np.sqrt(len(state_split) // 6))
            dim_2 = dim**2
            
            s = ''.join([U_dict[f] for f in state_split])
            slist = [s[:dim_2], s[2*dim_2:3*dim_2], s[dim_2:2*dim_2], s[5*dim_2:], s[4*dim_2:5*dim_2], s[3*dim_2:4*dim_2]]
            if force3:
                surf4 = ""
                for i in range(6):
                    surf4 += slist[i][0:2]
                    surf4 += slist[i][dim-1:dim+2]
                    surf4 += slist[i][dim_2-dim-1:dim_2-dim+2]
                    surf4 += slist[i][-1]
                return surf4

            if not even:
                return s[:dim_2] + s[2*dim_2:3*dim_2] + s[dim_2:2*dim_2] + s[5*dim_2:] + s[4*dim_2:5*dim_2] + s[3*dim_2:4*dim_2]                        
            surf4 = ""
            for i in range(6):
                surf4 += slist[i][0:3]
                surf4 += slist[i][dim-1:dim+3]
                surf4 += slist[i][dim*2-1:dim*2+3]
                surf4 += slist[i][dim_2-dim-1:dim_2-dim+3]
                surf4 += slist[i][-1]
            return surf4



        def translate(move, cubesize):
            rev = False
            rot = 1
            match = pattern.match(move)
            if match:
                # print("Input:", test_string)
                number = int(match.group(1)) if match.group(1) else 1
                letter = match.group(2)
                if bool(match.group(3)) and number == 1:
                    number = 2
                if match.group(4) == "'":
                    rot = -1
                elif match.group(4) == '2':
                    rot = 2
            else:
                print("Invalid input:", move)
            buff = 0
            if letter == "B":
                letter = "F"
                buff = cubesize - number
                rot = -rot
            elif letter == "L":
                letter = "R"
                buff = cubesize - number
                rot = -rot
            elif letter == "U":
                letter = "D"
                buff = cubesize - number
                rot = -rot
            out = ""
            for i in range(number):
                for j in range(abs(rot)):
                    if rot < 0:
                        out+="-"
                    out += letter.lower()
                    out += str(i+buff)
                    out += "."
            return out

        def get_sln(sln, cubesize) -> List[str]:
            cmds = sln.split(" ")
            out = ""
            for cmd in cmds:
                out += translate(cmd, cubesize)
            return out[:len(out)-1]

        dim = int(np.sqrt(len(state) // 6))
        even = (dim % 2 == 0)
        if force_odd:
            even = False

        state_for_solver = state2ubl(state, even, force3)
        # print(state_for_solver)
        cwd = os.getcwd()
        if not even or force3: # odd
            
            os.chdir(solver_path)
            result = subprocess.run(["python", "rubiks-cube-solver.py", "--state", state_for_solver], capture_output=True, text=True)
            os.chdir(cwd)
            raw_cmd = result.stdout
            sln = raw_cmd.split("Solution: ")
            sln = sln[1]    
            out = get_sln(sln, dim).split(".")
            return out
        else:
            os.chdir(solver_path_444)
            result = subprocess.run(["java", "-cp", ".:threephase.jar:twophase.jar", "solver", state_for_solver], capture_output=True, text=True)
            os.chdir(cwd)
            print(result)
            sln = result.stdout.split("OK")[1].strip().split(" ")
            out = ""

            for s in sln:
                rot = 1
                if "x" in s or "y" in s or "z" in s:
                    continue
                if "w" in s:
                    s = s.replace("w", "")
                    s = str(dim // 2) + s
                if s == "":
                    continue
                match = pattern.match(s)
                if match:
                    number = int(match.group(1)) if match.group(1) else 1
                    letter = match.group(2)
                    if bool(match.group(3)) and number == 1:
                        number = 2
                    if match.group(4) == "'":
                        rot = -1
                    elif match.group(4) == '2':
                        rot = 2
                else:
                    print(s)
                    continue
                buff = 0
                if letter == "B":
                    letter = "F"
                    buff = dim - number
                    rot = -rot
                elif letter == "L":
                    letter = "R"
                    buff = dim - number
                    rot = -rot
                elif letter == "U":
                    letter = "D"
                    buff = dim - number
                    rot = -rot
                for i in range(number):
                    for j in range(abs(rot)):
                        if rot < 0:
                            out+="-"
                        out += letter.lower()
                        out += str(i+buff)
                        out += "."
            print(out)
            return out[:len(out)-1].split(".")



def print_surface(state:Union[str, List[str]], cube_size:int, idx:int) -> None:
    """
    指定した面をprintする
    """
    for i in range(cube_size):
        print(",".join(state[cube_size*cube_size * idx + i*cube_size: cube_size*cube_size * idx + (i+1)*cube_size]))

def get_edge_indice(cube_size:int) -> List[List[Tuple[int]]]:
    """
    edgeのindiceを取得する。out[i][j][0,1]でi番目(0-11)のedgeのj番目(cube_size-2)の要素となる。intではなくtuple[int]なのはedgeが2つの面を持つから
    """
    edge_bias = {
        # 36,37,38, 0,1,2
        0: [cube_size * cube_size * 4, 0],
        1: [cube_size - 1, cube_size * cube_size * 2 + cube_size - 1],
        2: [cube_size * cube_size * 3 - 1, cube_size * cube_size * 6 - 1],
        3: [cube_size * cube_size * 6  - cube_size, cube_size * cube_size * 5 - cube_size],

        # 27,28,29, 2,1,0
        4: [cube_size * cube_size * 3, cube_size-1],
        5: [cube_size * cube_size - 1, cube_size * cube_size + cube_size - 1],
        6: [cube_size * cube_size * 2 - 1, cube_size * cube_size * 5 + cube_size - 1],
        7: [cube_size * cube_size * 6 - 1, cube_size * cube_size * 4 - cube_size],

        # 36,39,42, 29,32,35
        8: [cube_size*cube_size*5-cube_size, cube_size*cube_size*4-1],
        9: [cube_size*cube_size*4-cube_size, cube_size*cube_size*3-1],
        10: [cube_size*cube_size*3-cube_size, cube_size*cube_size*2-1],
        11: [cube_size*cube_size*2-cube_size, cube_size*cube_size*5-1],

    }

    edge_step = {
        # 36,37,38, 0,1,2
        0: [1, cube_size],
        1: [cube_size, -1],
        2: [-1, -cube_size],
        3: [-cube_size, 1],

        # 27,28,29, 2,1,0
        4: [1, -1],
        5: [-1, -1],
        6: [-1, -1],
        7: [-1, 1],

        # 36,39,42, 29,32,35
        8 : [-cube_size, -cube_size],
        9 : [-cube_size, -cube_size],
        10 : [-cube_size, -cube_size],
        11 : [-cube_size, -cube_size],
    }

    output = []

    for eid in range(12):
        tmp = []
        for i in range(cube_size - 2):
            tmp.append(
                tuple(edge_bias[eid][ee] + edge_step[eid][ee] * (i+1) for ee in range(2))
            )
        output.append(tmp)
    return output


def is_edge_complete(state:Union[str, List[str]], cube_size:int, edge_idx:int) -> bool:
    """
    edgeが完成しているかを判別する
    """
    edge = get_edge_indice(cube_size)[edge_idx]
    color = [state[edge[0][0]], state[edge[0][1]]]
    for e in edge:
        if state[e[0]] != color[0] or state[e[1]] != color[1]:
            return False
    return True


def count_complete_edge(state:str, cube_size:int) -> int:
    """
    完成したエッジの数を返す。edge最適化では10になったら終了
    """
    out = 0
    for i in range(12):
        if is_edge_complete(state, cube_size, i):
            out += 1
    return out


def print_edge(state:Union[str, List[str]], cube_size:int) -> None:
    """
    edgeの形状をprintする
    """
    indice = get_edge_indice(cube_size)
    for j in range(cube_size-2):
        outline = ""
        for i in range(12):
            if is_edge_complete(state, cube_size, i):
                outline += "*" + state[indice[i][j][0]] +state[indice[i][j][1]]
            else:
                outline += state[indice[i][j][0]] + state[indice[i][j][1]]
            outline += " "
            if i % 4 == 3:
                outline += "|"
        print(outline)


def get_cube_alias(initial_state:str, final_state:str, moves:str) -> List[str]:
    """
    final_stateの最終型に対応させてinitial_stateの色を指定する
    """
    pass

def translate_move(moves, cube_size):
    """
    独自定義のアクションを処理する。例えば d*はd{cube_size-1}に等しい、d0xは[d0, d0]に等しい
    """
    out = []
    for move in moves:
        if move == "":
            continue
        m = move.replace("*", str(cube_size-1))
        if "x" in move:
            # doubleの処理
            out.append(m.replace("x", ""))
            out.append(m.replace("x", ""))
        else:
            out.append(m)
    return out

def run_moves(state_in, moves, legal_moves, current_perm):
    """
    指定した移動を行う
    """
    perm = [c for c in current_perm]
    state = [s for s in state_in]
    for move in moves:
        mv = legal_moves[move]
        state = tuple(state[i] for i in mv)
        perm = tuple(perm[i] for i in mv)
    return state, perm


#### constants ####

"""
平面移動のindexの情報を保持。cubeの座標系を回転させて、移動先の面を上、移動元の面を下、面上のidxは左上スタートとなるようにする

t00, t01, t02..
t10, ...
tn0, tn1, ...

i00, i01, i02..
i10, ...
in0, in1, ...

こんな並び。keyの1-2はidx-1の面からidx-2の面へとキューブを動かすことに相当

"""

swap_act_dict = {
    "1-2" : [
        "d",
        [[0, -1], [1, 0]],
        [[0, -1], [1, 0]],
        True,
    ],
    "2-3" : [
        "d",
        [[0, -1], [1, 0]],
        [[0, -1], [1, 0]],
        True,
    ],
    "3-4" : [
        "d",
        [[0, -1], [1, 0]],
        [[0, -1], [1, 0]],
        True,
    ],
    "4-1" : [
        "d",
        [[0, -1], [1, 0]],
        [[0, -1], [1, 0]],
        True,
    ],

    "4-0" : [
        "f",
        [[1, 0], [0, 1]],
        [[0, -1], [1, 0]],
        True,
    ],
    "0-2" : [
        "f",
        [[0, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        True,
    ],
    "2-5" : [
        "f",
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
        True,
    ],
    "5-4" : [
        "f",
        [[0, 1], [-1, 0]],
        [[1, 0], [0, 1]],
        True,
    ],

    "1-0" : [
        "r",
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        True,
    ],
    "0-3" : [
        "r",
        [[-1, 0], [0, 1]],
        [[1, 0], [0, -1]],
        False,
    ],
    "3-5" : [
        "r",
        [[-1, 0], [0, -1]],
        [[1, 0], [0, 1]],
        True,
    ],
    "5-1" : [
        "r",
        [[-1, 0], [0, 1]],
        [[-1, 0], [0, 1]],
        False,
    ],
    "0-5" : [
        "-rx",
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        True,
    ],
    "2-4" : [
        "fx",
        [[-1, 0], [0, -1]],
        [[1, 0], [0, 1]],
        True,
    ],
    "1-3" : [
        "-rx",
        [[-1, 0], [0, -1]],
        [[1, 0], [0, 1]],
        False,
    ]

}


# 上述の条件で座標を再設定した際に、tgtの面で半時計回しにキューブを回すコマンド

surf_counter_clock = {
    0: "d*",
    1 : "-f0",
    2: "-r0",
    3 : "-f*",
    4 : "r*",
    5 : "d0",
    "1-3" : "f*",
    "0-5" : "-d0",
    "2-3" : "f*",
    "2-5" : "-d0",
    "3-0" : "-d*",
    "3-5" : "-d0",
    "4-3" : "f*",
    "4-5" : "-d0",
    "5-1" : "f0",
    "5-3" : "f*",
}


# 足りてない係数は機械的に生成
def get_inv(mat):
    if mat[0][0] != 0:
        return mat
    else:
        return [
            [-mat[0][0], -mat[0][1]],
            [-mat[1][0], -mat[1][1]],
        ]

keys = []
for key in swap_act_dict:
    keys.append(key)

for key in keys:
    swap_act_dict[key[2]+"-"+key[0]] = [
        get_rev(swap_act_dict[key][0]),
        [[ -swap_act_dict[key][2][0][0], -swap_act_dict[key][2][0][1] ], [-swap_act_dict[key][2][1][0], -swap_act_dict[key][2][1][1]]],
        [[ -swap_act_dict[key][1][0][0], -swap_act_dict[key][1][0][1] ], [-swap_act_dict[key][1][1][0], -swap_act_dict[key][1][1][1]]],
        not swap_act_dict[key][3]
    ]

swap_act_dict_inv = {}

for key in swap_act_dict:
    swap_act_dict_inv[key] = [
        swap_act_dict[key][0],
        get_inv(swap_act_dict[key][1]),
        get_inv(swap_act_dict[key][2]),
        swap_act_dict[key][3],
    ]


### edge処理 ###

"""
edge移動用のconstant。edge移動は本当はどこでもできるのだがidx-8のedgeでのみ行うようにしている。
手数が若干無駄になるが全体から見ると誤差なので無視

edge処理は手数全体から見た割合が少ないので最適化の優先度が低い
"""

edge_rotate_info = {
    # key番目のedgeのidx-0を変えるのがどのアクションかを返す
    # edgeを揃える作業は9-12のindice以外でやる予定がない
    8 : "-d",
    9 : "-d",
    10 : "-d",
    11 : "-d",
}

swap_tgt_moves = {
    8 : [["f0"], ["r0"]]
}


# 指定したidxのedgeを指定したidxに持ってくるコマンド
# 10-3はidx3のedgeをidx10に持っていく
swap_sac_moves = {'10-3': ['f0', 'd0', '-f0'], '10-2': ['f0', '-d0', '-f0'], '11-0': ['f0', 'd*', '-f0'], '11-1': ['f0', '-d*', '-f0'], '9-2': ['f*', 'd0', '-f*'], '9-3': ['f*', '-d0', '-f*'], '8-1': ['f*', 'd*', '-f*'], '8-0': ['f*', '-d*', '-f*'], '9-6': ['r0', 'd0', '-r0'], '9-7': ['r0', '-d0', '-r0'], '10-5': ['r0', 'd*', '-r0'], '10-4': ['r0', '-d*', '-r0'], '8-7': ['r*', 'd0', '-r*'], '8-6': ['r*', '-d0', '-r*'], '11-4': ['r*', 'd*', '-r*'], '11-5': ['r*', '-d*', '-r*'], '11-3': ['-f0', 'd0', 'f0'], '11-2': ['-f0', '-d0', 'f0'], '10-0': ['-f0', 'd*', 'f0'], '10-1': ['-f0', '-d*', 'f0'], '8-2': ['-f*', 'd0', 'f*'], '8-3': ['-f*', '-d0', 'f*'], '9-1': ['-f*', 'd*', 'f*'], '9-0': ['-f*', '-d*', 'f*'], '10-6': ['-r0', 'd0', 'r0'], '10-7': ['-r0', '-d0', 'r0'], '9-5': ['-r0', 'd*', 'r0'], '9-4': ['-r0', '-d*', 'r0'], '11-7': ['-r*', 'd0', 'r*'], '11-6': ['-r*', '-d0', 'r*'], '8-4': ['-r*', 'd*', 'r*'], '8-5': ['-r*', '-d*', 'r*']}

# パリティ除去のコマンド5x5x5のslnで知られている解法をやってるだけなので基本見なくて良い
parity_dict = {(0, 1, 2, 3): [], (2, 1, 3, 0): ['3a'], (0, 3, 1, 2): ['3b'], (1, 2, 0, 3): ['3c'], (3, 0, 2, 1): ['3d'], (1, 0, 2, 3): ['p1'], (0, 1, 3, 2): ['p2'], (3, 1, 0, 2): ['3a', '3a'], (1, 3, 2, 0): ['3a', '3b'], (0, 2, 3, 1): ['3a', '3c'], (2, 0, 1, 3): ['3a', '3d'], (2, 0, 3, 1): ['3a', 'p1'], (3, 1, 2, 0): ['3a', 'p2'], (1, 3, 0, 2): ['3b', 'p1'], (0, 2, 1, 3): ['3b', 'p2'], (1, 2, 3, 0): ['p1', '3a'], (3, 0, 1, 2): ['p1', '3b'], (2, 1, 0, 3): ['p1', '3c'], (0, 3, 2, 1): ['p1', '3d'], (1, 0, 3, 2): ['p1', 'p2'], (2, 3, 0, 1): ['3a', '3a', '3b'], (3, 2, 1, 0): ['3a', '3a', '3c'], (3, 2, 0, 1): ['3a', 'p1', '3a'], (2, 3, 1, 0): ['3a', 'p1', '3d']}

parity_cmd_dict = {
    "8-9" : {
        "x" : "-d",
        "l" : "d0",
        "r" : "-d*",
        "u" : "r0",
        "d" : "-r*",
        "f" : "-f*",
        "y" : "-d",

        "-x" : "d",
        "-l" : "-d0",
        "-r" : "d*",
        "-u" : "-r0",
        "-d" : "r*",
        "-f" : "f*",
        "-y" : "d",
    }
}

parity_move_dict = {
    # ややこしいから上下の反転を直す。4手無駄になるがまあ誤差
    "ab" : ["x", "u", "l", "-u", "f", "u", "-f", "-u", "-x", "-f", "-l", "f", "-u"],
    "cd" : ["-x", "-d", "-l", "d", "-f", "-d", "f", "d", "x", "f", "l", "-f", "d"],
    "p2" : ["x", "r", "u", "u", "x", "r", "f", "f", "x", "r", "f", "f", "-x", "-r", "f", "f", "-y", "l", "f", "f", "y", "-l", "u", "u", "r", "x", "u", "u", "-r", "-x", "u", "u", "-r", "-x"],
    "p1" : ["-x", "-r", "d", "d", "-x", "-r", "f", "f", "-x", "-r", "f", "f", "x", "r", "f", "f", "y", "-l", "f", "f", "-y", "l", "d", "d", "-r", "-x", "d", "d", "r", "x", "d", "d", "r", "x"],
}

### center-cubeを揃えるための関数 あまり効率は良くないが、全てのcenterをだいたい揃えてくれる ###

def get_center(state:Union[str, List[str]], cube_size:int, idx:int, center_dict:Optional[Dict[str, List[str]]]=None) -> Union[str, List[str]]:
    """
    指定したidxの面のcenterを表示する。絵柄を厳密に揃える系のタスクの場合はcenterの文字から各面が満たすべき色を出力する
    """
    if cube_size % 2 == 1:
        center_char = state[cube_size*cube_size * idx + (cube_size*cube_size)//2]
    else:
        center_char = state[cube_size*cube_size * idx + (cube_size*cube_size)//2 - cube_size // 2]
    if center_dict is not None:
        return center_dict[center_char]
    return center_char

def count_char(state:str, cube_size:int, idx:int, tgt:Union[str, List[str]], no_recur:bool=False)->int:
    """
    stateに対して、指定した面のidにおける、tgtと同じ色のタイルの数を列挙する
    完成が近い面を先に完成させたほうが手数が得なことが多いため
    """
    score = 0
    cnt = 0
    if isinstance(tgt, list) and not no_recur:
        idx_alias_0 = [i for i in range((cube_size-2)**2)]
        idx_alias_1 = [(cube_size-2)**2 - 1 - i for i in range((cube_size-2)**2)]
        idx_alias_2 = [ (cube_size - 2) ** 2 - (cube_size - 2) + (i // (cube_size - 2)) - (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]
        idx_alias_3 = [ (cube_size - 2) - 1 - (i // (cube_size - 2)) + (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]

        maxv = -1
        max_alias = None
        for alias in [idx_alias_0, idx_alias_1, idx_alias_2, idx_alias_3]:
            v = count_char(state, cube_size, idx, [tgt[idx] for idx in alias], True)
            # print(v, alias)
            if v > maxv:
                maxv = v
                max_alias = alias
        return maxv

    
    for y in range(cube_size):
        if y == 0 or y == cube_size - 1:
            continue
        for x in range(cube_size):
            if x == 0 or x == cube_size - 1:
                continue
            if isinstance(tgt, str):
                if state[idx*cube_size*cube_size+y*cube_size+x] == tgt:
                    score += 1
            else:
                if state[idx*cube_size*cube_size+y*cube_size+x] == tgt[cnt]:
                    score += 1
            cnt += 1
    return score


def get_target_with_img(state:List[str], final:List[str], cube_size:int, idx:int, print_trial:bool=False) -> List[str]:

    # 面の中央ごとにゴールとなる面の情報を取得
    center_dict = get_center_dict(final)
    # 現状のstateの指定したidxのcenterを取り出す
    key = get_center(state, cube_size, idx)
    # centerの情報をkeyにその面が満たすべきゴールを取得する
    center_char = center_dict[key]

    # 元の一致率が最も高いrotationで解釈する
    if isinstance(center_char, list):
        idx_alias_0 = [i for i in range((cube_size-2)**2)]
        idx_alias_1 = [(cube_size-2)**2 - 1 - i for i in range((cube_size-2)**2)]
        idx_alias_2 = [ (cube_size - 2) ** 2 - (cube_size - 2) + (i // (cube_size - 2)) - (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]
        idx_alias_3 = [ (cube_size - 2) - 1 - (i // (cube_size - 2)) + (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]

        maxv = -1
        max_alias = None
        if print_trial:
            print_surface(state, cube_size, idx)
    
        for alias in [idx_alias_0, idx_alias_1, idx_alias_2, idx_alias_3]:
            v = count_char(state, cube_size, idx, [center_char[idx] for idx in alias], True)
            if v > maxv:
                maxv = v
                max_alias = alias
            if print_trial:
                print([center_char[idx] for idx in alias], v)
            

    return [center_char[idx] for idx in max_alias]
    

def get_center_swap_mask_with_img(state:List[str], cube_size:int, surf_to:int, surf_from:int, final:List[str], print_mask=False) -> List[Tuple[int, int]]:
    """
    ある面から別の面にcenter-swap可能なマスのリストを表示する。座標系はswap_act_dictに準拠
    こちらは絵柄を考慮する
    """
    swap_info = swap_act_dict_inv[f"{surf_from}-{surf_to}"]
    
    mask = []
    
    target = get_target_with_img(state, final, cube_size, surf_to, print_mask)

    frm_surface = ""
    to_surface = ""
    tgt_surface = ""

    for y in range(cube_size):
        if y == 0 or y == cube_size - 1:
            continue
        for x in range(cube_size):
            if x == 0 or x == cube_size - 1:
                continue
            frm_x = swap_info[1][0][0] * x + swap_info[1][1][0] * y 
            frm_y = swap_info[1][0][1] * x + swap_info[1][1][1] * y 
            to_x = swap_info[2][0][0] * x + swap_info[2][1][0] * y 
            to_y = swap_info[2][0][1] * x + swap_info[2][1][1] * y 
            if frm_x < 0:
                frm_x += cube_size - 1
            if frm_y < 0:
                frm_y += cube_size - 1

            if to_x < 0:
                to_x += cube_size - 1
            if to_y < 0:
                to_y += cube_size - 1

            frm_char = state[cube_size*cube_size*surf_from + frm_x + frm_y * cube_size]
            to_char = state[cube_size*cube_size*surf_to + to_x + to_y * cube_size]
            frm_surface += frm_char
            to_surface += to_char
            tgt_idx = to_x -1 + (to_y - 1) * (cube_size - 2)
            tgt_surface += target[tgt_idx]
            if frm_char == target[tgt_idx] or (to_char != target[tgt_idx] and to_char in target):
                mask.append((x, y))
                frm_surface += "+"
                to_surface += "+"
                tgt_surface += "+"
            frm_surface += ","
            to_surface += ","
            tgt_surface += ","
        frm_surface += "\n"
        to_surface += "\n"
        tgt_surface += "\n"

    # debug
    if print_mask:
        print(frm_surface)
        print(to_surface)
        print(tgt_surface)
        print(mask)
        
    #raise ValueError

    return mask



def get_center_swap_mask(state:str, cube_size:int, surf_to:int, surf_from:int, center_char:Union[str, List[str]]) -> List[Tuple[int, int]]:
    """
    ある面から別の面にcenter-swap可能なマスのリストを表示する。座標系はswap_act_dictに準拠
    """
    swap_info = swap_act_dict_inv[f"{surf_from}-{surf_to}"]
    
    mask = []
    cnt = 0

    # 元の一致率が最も高いrotationで解釈する
    if isinstance(center_char, list):
        idx_alias_0 = [i for i in range((cube_size-2)**2)]
        idx_alias_1 = [(cube_size-2)**2 - 1 - i for i in range((cube_size-2)**2)]
        idx_alias_2 = [ (cube_size - 2) ** 2 - (cube_size - 2) + (i // (cube_size - 2)) - (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]
        idx_alias_3 = [ (cube_size - 2) - 1 - (i // (cube_size - 2)) + (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]

        maxv = -1
        max_alias = None
        for alias in [idx_alias_0, idx_alias_1, idx_alias_2, idx_alias_3]:
            #print(alias)
            #print(center_char)
            v = count_char(state, cube_size, surf_to, [center_char[idx] for idx in alias], True)
            if v > maxv:
                maxv = v
                max_alias = alias

        rev_alias = [-1 for _ in range(len(max_alias))]
        for i in range(len(max_alias)):
            rev_alias[max_alias[i]] = i

    
    for y in range(cube_size):
        if y == 0 or y == cube_size - 1:
            continue
        for x in range(cube_size):
            if x == 0 or x == cube_size - 1:
                continue
            frm_x = swap_info[1][0][0] * x + swap_info[1][1][0] * y 
            frm_y = swap_info[1][0][1] * x + swap_info[1][1][1] * y 
            to_x = swap_info[2][0][0] * x + swap_info[2][1][0] * y 
            to_y = swap_info[2][0][1] * x + swap_info[2][1][1] * y 
            if frm_x < 0:
                frm_x += cube_size - 1
            if frm_y < 0:
                frm_y += cube_size - 1

            if to_x < 0:
                to_x += cube_size - 1
            if to_y < 0:
                to_y += cube_size - 1

            frm_char = state[cube_size*cube_size*surf_from + frm_x + frm_y * cube_size]
            to_char = state[cube_size*cube_size*surf_to + to_x + to_y * cube_size]
            #print(frm_char, to_char)
            if isinstance(center_char, list):
                if frm_char == center_char[rev_alias[cnt]]:
                    print(frm_char, to_char)
                    mask.append((x, y))
            else:
                if frm_char == center_char and to_char != center_char:
                    mask.append((x, y))
            cnt += 1
    return mask


def find_max_subset(coords):
    # xの値でソート
    sorted_coords = sorted(coords, key=lambda x: x[0])
    
    # 共通のxを持つ最大の部分集合を見つける
    max_subset = []
    current_subset = [sorted_coords[0]]

    for i in range(1, len(sorted_coords)):
        if sorted_coords[i][0] == current_subset[-1][0]:
            # xが共通している場合、部分集合に追加
            current_subset.append(sorted_coords[i])
        else:
            # xが共通していない場合、新しい部分集合を開始
            if len(current_subset) > len(max_subset):
                max_subset = current_subset
            current_subset = [sorted_coords[i]]

    # 最後の部分集合を確認
    if len(current_subset) > len(max_subset):
        max_subset = current_subset

    return max_subset

def get_swap_act(surf_to:int, surf_from:int) -> Tuple[str]:
    return swap_act_dict[f"{surf_from}-{surf_to}"][0], surf_counter_clock[surf_to] if f"{surf_from}-{surf_to}" not in surf_counter_clock else surf_counter_clock[f"{surf_from}-{surf_to}"]


def swap_to_surface(surf_to:int, surf_from:int, masks:List[Tuple[int, int]], cube_size:int) -> List[str]:
    """
    centerを揃えるためのパーツの移動。以下の条件を満たす場合、maskは同時に移動させることができる
    ・maskのxのuniqueな値の集合と、y or size-yの集合に重複がない
    ・maskを構成する任意のxと任意のyの組み合わせから得られる座標がmaskの中に含まれている

    surfaceの回転処理が死ぬほどややこしくてゲロ吐きそう
    """
    act_surf, act_rotate = get_swap_act(surf_to, surf_from)
    output = []
    x_set = set()
    for m in masks:
        x_set.add(m[0])
    
    y_set = set()
    for m in masks:
        if m[1] not in x_set:
            y_set.add(m[1])
    
    y_set_rev = set()
    for m in masks:
        if cube_size - 1 - m[1] not in x_set:
            y_set_rev.add(cube_size-1 - m[1])
    
    if len(y_set_rev) > len(y_set):    
        y_set = y_set_rev
        act_rotate = get_rev(act_rotate)

    for x in x_set:
        x_use = x
        if swap_act_dict[f"{surf_from}-{surf_to}"][3]:
            x_use = cube_size - 1 - x
        output.append(f"{act_surf}{x_use}")
    
    output.append(act_rotate)

    for y in y_set:
        y_use = y
        if swap_act_dict[f"{surf_from}-{surf_to}"][3]:
            y_use = cube_size - 1 - y
        output.append(f"{act_surf}{y_use}")

    output.append(get_rev(act_rotate))

    for x in x_set:
        if swap_act_dict[f"{surf_from}-{surf_to}"][3]:
            x_use = cube_size - 1 - x
        output.append(f"{get_rev(act_surf)}{x_use}")

    output.append(act_rotate)

    for y in y_set:
        y_use = y
        if swap_act_dict[f"{surf_from}-{surf_to}"][3]:
            y_use = cube_size - 1 - y
        output.append(f"{get_rev(act_surf)}{y_use}")

    return output, len(y_set)

def get_center_dict(final_state:List[str]) -> Dict[str, List[str]]:
    dim = int(np.sqrt(len(final_state) // 6))
    out = {}
    for i in range(6):
        key = get_center(final_state, dim, i)
        out[key] = []

        cnt = 0
        for y in range(dim):
            if y == 0 or y == dim - 1:
                continue
            for x in range(dim):
                if x == 0 or x == dim - 1:
                    continue
                out[key].append(final_state[dim*dim*i + y*dim + x])
    return out

### center-cubeを揃えるための関数 ここまで ###


### edgeの処理 ###

def get_edge_color(state:str, cube_size:int, edge_idx:int, even_parity=False) -> tuple[str, str]:
    """
    指定したstateの指定したidxのedgeの揃えるべき色を返す
    """
    edge = get_edge_indice(cube_size)[edge_idx]
    #print(edge)
    cedge = edge[(len(edge)) // 2]
    #print(cedge)
    if even_parity and cube_size % 2 == 0:
        return [state[cedge[1]], state[cedge[0]]]

    return [state[cedge[0]], state[cedge[1]]]


def get_sacrifice_idx(state:str, cube_size:int, idx:int) -> Optional[int]:
    """
    idxのedgeを揃える際に犠牲にするidxを返す。犠牲にできるものがないときはNoneを返す
    """
    bidx = idx // 4
    for i in range(12):
        if i // 4 == bidx:
            continue
        if not is_edge_complete(state, cube_size, i):
            return i
    return None

def get_edge_change_cmd_list(state:str, cube_size:int, idx:int) -> Tuple[List[str], bool]:
    # 指定したedgeの近くにある交換可能なedgeのidxと交換に必要なアクションを返す
    assert idx // 4 == 2
    rotate_act = edge_rotate_info[idx]
    rev_rm_act = None
    tgt_indice, has_swap = find_edge_target_indice(state,cube_size, idx)
    
    out = []
    # TODO: idxの反転
    for eid in tgt_indice[0]:
        out.append(f"{get_rev(rotate_act)}{eid+1}")
        rev_rm_act = get_rev(rotate_act)

    for eid in tgt_indice[1]:
        out.append(f"{rotate_act}{eid+1}")
        rev_rm_act = rotate_act

    for eid in tgt_indice[2]:
        out.append(f"{rotate_act}{eid+1}x")
        if rev_rm_act is None:
            rev_rm_act = rotate_act + "x"

    if rev_rm_act is None:
        rev_rm_act = rotate_act

    for eid in tgt_indice[3]:
        out.append(f"{rev_rm_act}{eid+1}")

    return out, has_swap

def find_edge_target_indice(state:str, cube_size:int, idx:int) -> Tuple[List[List[int]], bool]:
    """
    回転によってedgeを転送できる候補を探す.
    out[i][j]はi番目の回転でedgeを揃えられるj番目のデータ
    i=0番目が時計回り、1番目が半時計回り、2番目は半回転, 3番目は自分自身の反転除去
    もう一つの返り値は上下が反転したedgeが一つでも存在するかを返す。これがTrueだと上下を反転させた後にedge揃えをもう一度やる必要がある
    """
    assert idx // 4  == 2
    idx_4 = idx % 4
    idx_bias = (idx // 4) * 4
    edges = get_edge_indice(cube_size)
    tgt_color = get_edge_color(state, cube_size, idx)
    #print("tgt", tgt_color)
    rot_set = set() # for debug and self rotate
    output = []
    has_swap = False

    for rotate in [1, -1, 2]:
        tmp = []
        tgt_idx = idx_4 + rotate
        if tgt_idx < 0:
            tgt_idx += 4
        tgt_idx = tgt_idx % 4

        tgt_idx = idx_bias + tgt_idx

        for i in range(cube_size-2):
            # print(state[edges[tgt_idx][i][0]]+state[edges[tgt_idx][i][1]])
            if state[edges[tgt_idx][i][0]] == tgt_color[0] and state[edges[tgt_idx][i][1]] == tgt_color[1]:
                tmp.append(i)
                assert i not in rot_set
                rot_set.add(i)
            # 上下反転
            if state[edges[tgt_idx][i][0]] == tgt_color[1] and state[edges[tgt_idx][i][1]] == tgt_color[0]:
                has_swap = True
        output.append(tmp)

    # 全ての処理が終わった後にtgt自身に反転したやつがないかを確かめる
    rtmp = []
    for i in range(cube_size-2):
        if state[edges[idx][i][0]] == tgt_color[1] and state[edges[idx][i][1]] == tgt_color[0]:
            if i not in rot_set:
                rtmp.append(i)
            has_swap = True
    output.append(rtmp)
    return output, has_swap
        

def get_preopt_list_for_edgeopt(idx:int) -> List[str]:
    """
    edgeoptの前に行えるmoveのリストを返す。preprocの長さは精々4でありedgeoptに比べると手数は十分少ないので、一番いいやつを探す
    """
    assert idx in swap_tgt_moves
    out = []
    for rot1 in ["", "d0", "-d0"]:
        for rot2 in ["", "d*", "-d*"]:
            for sw1 in ["", swap_tgt_moves[idx][0][0], get_rev(swap_tgt_moves[idx][0][0])]:
                for sw2 in ["", swap_tgt_moves[idx][1][0], get_rev(swap_tgt_moves[idx][1][0])]:
                    if sw1 == "" and sw2 == "":
                        if rot1 != "" or rot2 != "":
                            continue
                    out.append(
                        [
                            rot1, rot2, sw1, sw2
                        ]
                    )
    return out

def get_best_preopt_for_edgeopt(state:str, cube_size:int, idx:int, legal_moves, perm) -> List[str]:
    """
    edgeopt前にやる最適なmovesを返す
    """
    moves_candidate = get_preopt_list_for_edgeopt(idx)
    best_proc = []
    best_score = -1
    best_raw_score = []
    state_test = state
    for moves in moves_candidate:
        state_test = state
        perm_test = [p for p in perm]
        state_test, perm_test = run_moves(state_test, translate_move(moves, cube_size), legal_moves, perm_test)
        if get_sacrifice_idx(state_test, cube_size, idx) is None:
            continue
        score, swap = get_edge_change_cmd_list(state_test, cube_size, idx)
        #print(moves, len(score), score)
        #show_edge(state_test, cube_size)
        #print("============")
        
        if len(score) > best_score:
            best_proc = moves
            best_score = len(score)
            best_raw_score = score
        elif len(score) == 0 and best_score == 0 and swap:
            best_proc = moves
    print(best_score, best_proc, best_raw_score)
    # sac or swapの始末すらできないってのはないはず？
    assert best_score != -1
    return best_proc

def complete_edge(inisial_state:str, cube_size:int, idx:int, legal_moves: dict[tuple[int, ...], list[str]], perm) -> Tuple[str, List[str]]:
    assert idx in swap_tgt_moves
    state = inisial_state
    mvs = []
    print(get_edge_color(inisial_state, cube_size, idx))
    for _ in range(12):
        if is_edge_complete(state, cube_size, idx) or count_complete_edge(state, cube_size) >= 10:
            print("completed")
            return state, mvs
        preproc = get_best_preopt_for_edgeopt(state, cube_size, idx, legal_moves, perm)
        preproc = translate_move(preproc, cube_size)
        mvs.extend(preproc)
        state, perm = run_moves(state, preproc, legal_moves, perm)
        print_edge(state, cube_size)
        print("preproc done")
        print("----------------------")
        sac_idx = get_sacrifice_idx(state, cube_size, idx)
        print(sac_idx, "sacidx")

        ecmoves, swap = get_edge_change_cmd_list(state, cube_size, idx)
        swap_move = []

        # swapしたやつしか残ってない場合
        if len(ecmoves) == 0 and swap:
            print("search swap")
            for smv in swap_tgt_moves[idx]:
                test_state = state
                test_perm = [p for p in perm]
                swp = [smv[0], smv[0]]
                test_state, test_perm = run_moves(test_state, translate_move(swp, cube_size), legal_moves, test_perm)
                ecmoves_test, _ = get_edge_change_cmd_list(test_state, cube_size, idx)
                if len(ecmoves_test) != 0:
                    print("found swap")
                    swap_move = swp
                    ecmoves.extend(ecmoves_test)
                    break

        moves = []
        moves.extend(swap_move)
        moves.extend(ecmoves)
        
        moves.extend(swap_sac_moves[f"{idx}-{sac_idx}"])
        moves.extend([get_rev(m) for m in ecmoves])
        moves.append(
            get_rev(swap_sac_moves[f"{idx}-{sac_idx}"][1])
        )
        moves.append(
            swap_sac_moves[f"{idx}-{sac_idx}"][2]
        )
        print(moves)
        swap_move = translate_move(moves, cube_size)
        mvs.extend(swap_move)
        state, perm = run_moves(state, swap_move, legal_moves, perm)
        print_edge(state, cube_size)
        print("*************************")
    raise ValueError


def remove_parity(state:str, cube_size:int, edge1:int, edge2:int, legal_moves, perm) -> Tuple[str, List[str]]:
    output_moves = []
    pmove_len = 0
    tmove_len = 0
    assert edge1 == 8 and edge2 == 9
    for edge in [edge1, edge2]:
        if is_edge_complete(state, cube_size, edge):
            for j in range(12):
                if j == edge1 or j == edge2:
                    continue
                if not is_edge_complete(state, cube_size, j):
                    if f"{edge}-{j}" in swap_sac_moves:
                        swap_moves = [m for m in swap_sac_moves[f"{edge}-{j}"]]
                        swap_moves = translate_move(swap_moves, cube_size)
                        state, perm = run_moves(state, swap_moves, legal_moves, perm)
                        output_moves.extend(swap_moves)
                    else:
                        swap_moves = [m for m in swap_sac_moves[f"{j}-{0}"]]
                        swap_moves = translate_move(swap_moves, cube_size)
                        state,perm = run_moves(state, swap_moves, legal_moves, perm)
                        output_moves.extend(swap_moves)
                        for k in range(8):
                            if not is_edge_complete(state, cube_size, k):
                                swap_moves2 = [m for m in swap_sac_moves[f"{edge}-{k}"]]
                                swap_moves2 = translate_move(swap_moves2, cube_size)
                                state, perm = run_moves(state, swap_moves2, legal_moves, perm)
                                output_moves.extend(swap_moves2)
                    break

    print_edge(state, cube_size)

    # raise ValueError

    center1 = get_edge_color(state, cube_size, edge1, even_parity=False)
    center2 = get_edge_color(state, cube_size, edge2, even_parity=False)
    indice = get_edge_indice(cube_size)
    indice1 = indice[edge1]
    indice2 = indice[edge2]
    
    parity_cmd = parity_cmd_dict[f"{edge1}-{edge2}"]
    for i in range((cube_size // 2)-1):
        cs = [indice1[i], indice1[-1-i], indice2[i], indice2[-1-i]]
        parities = []
        for j, c in enumerate(cs):
            print(state[c[0]],state[c[1]])
            if state[c[0]] == center1[0] and state[c[1]] == center1[1]:
                if j == 0:
                    parities.append(0)
                elif j == 1:
                    parities.append(1)
                elif j == 2:
                    parities.append(0)
                elif j == 3:
                    parities.append(1)
            elif state[c[0]] == center1[1] and state[c[1]] == center1[0]:
                if j == 0:
                    parities.append(1)
                elif j == 1:
                    parities.append(0)
                elif j == 2:
                    parities.append(1)
                elif j == 3:
                    parities.append(0)
                    
            if state[c[0]] == center2[0] and state[c[1]] == center2[1]:
                if j == 0:
                    parities.append(2)
                elif j == 1:
                    parities.append(3)
                elif j == 2:
                    parities.append(2)
                elif j == 3:
                    parities.append(3)
            elif state[c[0]] == center2[1] and state[c[1]] == center2[0]:
                if j == 0:
                    parities.append(3)
                elif j == 1:
                    parities.append(2)
                elif j == 2:
                    parities.append(3)
                elif j == 3:
                    parities.append(2)
        parities = tuple(parities)
        acts = parity_dict[parities]

        # test
        print(parities)
        acts = parity_dict[parities]
        print(acts)
        for act in acts:    
            if act == "3a":
                x = i + 1
                moves = [parity_cmd[m] if "x" not in m else parity_cmd[m] + str(x) for m in parity_move_dict["ab"]]
            elif act == "3b":
                x = cube_size - 2 - i
                moves = [parity_cmd[m] if "x" not in m else parity_cmd[m] + str(x) for m in parity_move_dict["ab"]]
            elif act == "3c":
                x = i + 1
                moves = [parity_cmd[m] if "x" not in m else parity_cmd[m] + str(x) for m in parity_move_dict["cd"]]
            elif act == "3d":
                x = cube_size - 2 - i
                moves = [parity_cmd[m] if "x" not in m else parity_cmd[m] + str(x) for m in parity_move_dict["cd"]]
            elif act == "p1":
                x = i + 1
                y = cube_size - 2 - i
                moves = [parity_cmd[m] + str(x) if "x" in m else parity_cmd[m] + str(y) if "y" in m else parity_cmd[m] for m in parity_move_dict["p1"]]

            elif act == "p2":
                x = i + 1
                y = cube_size - 2 - i
                moves = [parity_cmd[m] + str(x) if "x" in m else parity_cmd[m] + str(y) if "y" in m else parity_cmd[m] for m in parity_move_dict["p2"]]
                
            moves = translate_move(moves, cube_size)
            output_moves.extend(moves)
            print(moves)

            if act.startswith("p"):
                pmove_len += len(moves)
            else:
                tmove_len += len(moves)

            for mv in moves:
                state, perm = run_moves(state, [mv], legal_moves, perm)
            print_edge(state, cube_size)
            print("---------------")
    return state, output_moves, pmove_len, tmove_len



### edgeの処理ここまで ###

### 絵柄を塗り直す処理 ###

def get_alias(initial:str, moves:List[str], legal_moves) -> List[str]:
    perm = [i for i in range(len(initial))]
    _, perm = run_moves(initial, moves, legal_moves, perm)
    dim = int(np.sqrt(len(initial) // 6))
    output = ["" for i in range(len(initial))]
    surf = ["A", "B", "C", "D", "E", "F"]
    cnt = 0
    for i in range(6):
        for j in range(dim*dim):
            output[perm[cnt]] = surf[i]
            cnt += 1
    return output
    

### 絵柄を塗りつぶす処理ここまで ###

### 最後の面の絵柄を揃えるソルバ ###

# 面ごとの16手1組の表面の3点回転
#surf_perm_xy = {(2, 1, 3, 0): ['-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm', '-d*', '-d*', '-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm'], (1, 2, 0, 3): ['-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn', '-d*', '-d*', '-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn'], (0, 2, 3, 1): ['-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv', '-d*', '-d*', '-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv'], (1, 3, 2, 0): ['-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb', '-d*', '-d*', '-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb'], (3, 1, 0, 2): ['-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm', 'd*', 'd*', '-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm'], (2, 0, 1, 3): ['-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn', 'd*', 'd*', '-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn'], (0, 3, 1, 2): ['-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv', 'd*', 'd*', '-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv'], (3, 0, 2, 1): ['-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb', 'd*', 'd*', '-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb']}
#surf_perm_xx = {(2, 1, 3, 0): ['-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm', '-d*', '-d*', '-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm'], (0, 2, 3, 1): ['-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn', '-d*', '-d*', '-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn'], (1, 2, 0, 3): ['-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn', '-d*', '-d*', '-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn'], (1, 3, 2, 0): ['-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm', '-d*', '-d*', '-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm'], (3, 1, 0, 2): ['-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm', 'd*', 'd*', '-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm'], (0, 3, 1, 2): ['-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn', 'd*', 'd*', '-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn'], (2, 0, 1, 3): ['-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn', 'd*', 'd*', '-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn'], (3, 0, 2, 1): ['-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm', 'd*', 'd*', '-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm']}

# surf_perm_xy = {(2, 1, 3, 0): ['-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm', '-d*', '-d*', '-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm'], (1, 2, 0, 3): ['-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn', '-d*', '-d*', '-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn'], (0, 2, 3, 1): ['-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv', '-d*', '-d*', '-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv'], (1, 3, 2, 0): ['-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb', '-d*', '-d*', '-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb'], (3, 1, 0, 2): ['-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm', 'd*', 'd*', '-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm'], (2, 0, 1, 3): ['-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn', 'd*', 'd*', '-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn'], (0, 3, 1, 2): ['-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv', 'd*', 'd*', '-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv'], (3, 0, 2, 1): ['-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb', 'd*', 'd*', '-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb']}
surf_perm_xy = [{(2, 1, 3, 0): ['-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm', '-d*', '-d*', '-rm', 'dv', 'rm', 'd*', '-rm', '-dv', 'rm'], (1, 2, 0, 3): ['-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn', '-d*', '-d*', '-rn', 'db', 'rn', 'd*', '-rn', '-db', 'rn'], (0, 2, 3, 1): ['-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv', '-d*', '-d*', '-rv', 'dn', 'rv', 'd*', '-rv', '-dn', 'rv'], (1, 3, 2, 0): ['-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb', '-d*', '-d*', '-rb', 'dm', 'rb', 'd*', '-rb', '-dm', 'rb'], (3, 1, 0, 2): ['-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm', 'd*', 'd*', '-rm', '-dv', 'rm', '-d*', '-rm', 'dv', 'rm'], (2, 0, 1, 3): ['-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn', 'd*', 'd*', '-rn', '-db', 'rn', '-d*', '-rn', 'db', 'rn'], (0, 3, 1, 2): ['-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv', 'd*', 'd*', '-rv', '-dn', 'rv', '-d*', '-rv', 'dn', 'rv'], (3, 0, 2, 1): ['-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb', 'd*', 'd*', '-rb', '-dm', 'rb', '-d*', '-rb', 'dm', 'rb']}, {(0, 3, 1, 2): ['-dn', 'fv', 'dn', 'f0', '-dn', '-fv', 'dn', '-f0', '-f0', '-dn', 'fv', 'dn', 'f0', '-dn', '-fv', 'dn'], (3, 0, 2, 1): ['-dm', 'fb', 'dm', 'f0', '-dm', '-fb', 'dm', '-f0', '-f0', '-dm', 'fb', 'dm', 'f0', '-dm', '-fb', 'dm'], (2, 0, 1, 3): ['-db', 'fn', 'db', 'f0', '-db', '-fn', 'db', '-f0', '-f0', '-db', 'fn', 'db', 'f0', '-db', '-fn', 'db'], (3, 1, 0, 2): ['-dv', 'fm', 'dv', 'f0', '-dv', '-fm', 'dv', '-f0', '-f0', '-dv', 'fm', 'dv', 'f0', '-dv', '-fm', 'dv'], (0, 2, 3, 1): ['-dn', '-fv', 'dn', '-f0', '-dn', 'fv', 'dn', 'f0', 'f0', '-dn', '-fv', 'dn', '-f0', '-dn', 'fv', 'dn'], (1, 3, 2, 0): ['-dm', '-fb', 'dm', '-f0', '-dm', 'fb', 'dm', 'f0', 'f0', '-dm', '-fb', 'dm', '-f0', '-dm', 'fb', 'dm'], (1, 2, 0, 3): ['-db', '-fn', 'db', '-f0', '-db', 'fn', 'db', 'f0', 'f0', '-db', '-fn', 'db', '-f0', '-db', 'fn', 'db'], (2, 1, 3, 0): ['-dv', '-fm', 'dv', '-f0', '-dv', 'fm', 'dv', 'f0', 'f0', '-dv', '-fm', 'dv', '-f0', '-dv', 'fm', 'dv']}, {(3, 0, 2, 1): ['-dm', 'rb', 'dm', 'r0', '-dm', '-rb', 'dm', '-r0', '-r0', '-dm', 'rb', 'dm', 'r0', '-dm', '-rb', 'dm'], (0, 3, 1, 2): ['-dn', 'rv', 'dn', 'r0', '-dn', '-rv', 'dn', '-r0', '-r0', '-dn', 'rv', 'dn', 'r0', '-dn', '-rv', 'dn'], (3, 1, 0, 2): ['-dv', 'rm', 'dv', 'r0', '-dv', '-rm', 'dv', '-r0', '-r0', '-dv', 'rm', 'dv', 'r0', '-dv', '-rm', 'dv'], (2, 0, 1, 3): ['-db', 'rn', 'db', 'r0', '-db', '-rn', 'db', '-r0', '-r0', '-db', 'rn', 'db', 'r0', '-db', '-rn', 'db'], (1, 3, 2, 0): ['-dm', '-rb', 'dm', '-r0', '-dm', 'rb', 'dm', 'r0', 'r0', '-dm', '-rb', 'dm', '-r0', '-dm', 'rb', 'dm'], (0, 2, 3, 1): ['-dn', '-rv', 'dn', '-r0', '-dn', 'rv', 'dn', 'r0', 'r0', '-dn', '-rv', 'dn', '-r0', '-dn', 'rv', 'dn'], (2, 1, 3, 0): ['-dv', '-rm', 'dv', '-r0', '-dv', 'rm', 'dv', 'r0', 'r0', '-dv', '-rm', 'dv', '-r0', '-dv', 'rm', 'dv'], (1, 2, 0, 3): ['-db', '-rn', 'db', '-r0', '-db', 'rn', 'db', 'r0', 'r0', '-db', '-rn', 'db', '-r0', '-db', 'rn', 'db']}, {(1, 3, 2, 0): ['-dm', 'fv', 'dm', 'f*', '-dm', '-fv', 'dm', '-f*', '-f*', '-dm', 'fv', 'dm', 'f*', '-dm', '-fv', 'dm'], (0, 2, 3, 1): ['-dn', 'fb', 'dn', 'f*', '-dn', '-fb', 'dn', '-f*', '-f*', '-dn', 'fb', 'dn', 'f*', '-dn', '-fb', 'dn'], (2, 1, 3, 0): ['-dv', 'fn', 'dv', 'f*', '-dv', '-fn', 'dv', '-f*', '-f*', '-dv', 'fn', 'dv', 'f*', '-dv', '-fn', 'dv'], (1, 2, 0, 3): ['-db', 'fm', 'db', 'f*', '-db', '-fm', 'db', '-f*', '-f*', '-db', 'fm', 'db', 'f*', '-db', '-fm', 'db'], (3, 0, 2, 1): ['-dm', '-fv', 'dm', '-f*', '-dm', 'fv', 'dm', 'f*', 'f*', '-dm', '-fv', 'dm', '-f*', '-dm', 'fv', 'dm'], (0, 3, 1, 2): ['-dn', '-fb', 'dn', '-f*', '-dn', 'fb', 'dn', 'f*', 'f*', '-dn', '-fb', 'dn', '-f*', '-dn', 'fb', 'dn'], (3, 1, 0, 2): ['-dv', '-fn', 'dv', '-f*', '-dv', 'fn', 'dv', 'f*', 'f*', '-dv', '-fn', 'dv', '-f*', '-dv', 'fn', 'dv'], (2, 0, 1, 3): ['-db', '-fm', 'db', '-f*', '-db', 'fm', 'db', 'f*', 'f*', '-db', '-fm', 'db', '-f*', '-db', 'fm', 'db']}, {(0, 2, 3, 1): ['-dn', 'rb', 'dn', 'r*', '-dn', '-rb', 'dn', '-r*', '-r*', '-dn', 'rb', 'dn', 'r*', '-dn', '-rb', 'dn'], (1, 3, 2, 0): ['-dm', 'rv', 'dm', 'r*', '-dm', '-rv', 'dm', '-r*', '-r*', '-dm', 'rv', 'dm', 'r*', '-dm', '-rv', 'dm'], (1, 2, 0, 3): ['-db', 'rm', 'db', 'r*', '-db', '-rm', 'db', '-r*', '-r*', '-db', 'rm', 'db', 'r*', '-db', '-rm', 'db'], (2, 1, 3, 0): ['-dv', 'rn', 'dv', 'r*', '-dv', '-rn', 'dv', '-r*', '-r*', '-dv', 'rn', 'dv', 'r*', '-dv', '-rn', 'dv'], (0, 3, 1, 2): ['-dn', '-rb', 'dn', '-r*', '-dn', 'rb', 'dn', 'r*', 'r*', '-dn', '-rb', 'dn', '-r*', '-dn', 'rb', 'dn'], (3, 0, 2, 1): ['-dm', '-rv', 'dm', '-r*', '-dm', 'rv', 'dm', 'r*', 'r*', '-dm', '-rv', 'dm', '-r*', '-dm', 'rv', 'dm'], (2, 0, 1, 3): ['-db', '-rm', 'db', '-r*', '-db', 'rm', 'db', 'r*', 'r*', '-db', '-rm', 'db', '-r*', '-db', 'rm', 'db'], (3, 1, 0, 2): ['-dv', '-rn', 'dv', '-r*', '-dv', 'rn', 'dv', 'r*', 'r*', '-dv', '-rn', 'dv', '-r*', '-dv', 'rn', 'dv']}, {(2, 0, 1, 3): ['-rn', 'dv', 'rn', 'd0', '-rn', '-dv', 'rn', '-d0', '-d0', '-rn', 'dv', 'rn', 'd0', '-rn', '-dv', 'rn'], (3, 1, 0, 2): ['-rm', 'db', 'rm', 'd0', '-rm', '-db', 'rm', '-d0', '-d0', '-rm', 'db', 'rm', 'd0', '-rm', '-db', 'rm'], (3, 0, 2, 1): ['-rb', 'dn', 'rb', 'd0', '-rb', '-dn', 'rb', '-d0', '-d0', '-rb', 'dn', 'rb', 'd0', '-rb', '-dn', 'rb'], (0, 3, 1, 2): ['-rv', 'dm', 'rv', 'd0', '-rv', '-dm', 'rv', '-d0', '-d0', '-rv', 'dm', 'rv', 'd0', '-rv', '-dm', 'rv'], (1, 2, 0, 3): ['-rn', '-dv', 'rn', '-d0', '-rn', 'dv', 'rn', 'd0', 'd0', '-rn', '-dv', 'rn', '-d0', '-rn', 'dv', 'rn'], (2, 1, 3, 0): ['-rm', '-db', 'rm', '-d0', '-rm', 'db', 'rm', 'd0', 'd0', '-rm', '-db', 'rm', '-d0', '-rm', 'db', 'rm'], (1, 3, 2, 0): ['-rb', '-dn', 'rb', '-d0', '-rb', 'dn', 'rb', 'd0', 'd0', '-rb', '-dn', 'rb', '-d0', '-rb', 'dn', 'rb'], (0, 2, 3, 1): ['-rv', '-dm', 'rv', '-d0', '-rv', 'dm', 'rv', 'd0', 'd0', '-rv', '-dm', 'rv', '-d0', '-rv', 'dm', 'rv']}]

# surf_perm_xx = {(2, 1, 3, 0): ['-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm', '-d*', '-d*', '-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm'], (0, 2, 3, 1): ['-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn', '-d*', '-d*', '-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn'], (1, 2, 0, 3): ['-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn', '-d*', '-d*', '-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn'], (1, 3, 2, 0): ['-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm', '-d*', '-d*', '-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm'], (3, 1, 0, 2): ['-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm', 'd*', 'd*', '-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm'], (0, 3, 1, 2): ['-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn', 'd*', 'd*', '-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn'], (2, 0, 1, 3): ['-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn', 'd*', 'd*', '-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn'], (3, 0, 2, 1): ['-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm', 'd*', 'd*', '-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm']}
surf_perm_xx = [{(2, 1, 3, 0): ['-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm', '-d*', '-d*', '-rm', 'dn', 'rm', 'd*', '-rm', '-dn', 'rm'], (0, 2, 3, 1): ['-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn', '-d*', '-d*', '-rn', 'dv', 'rn', 'd*', '-rn', '-dv', 'rn'], (1, 2, 0, 3): ['-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn', '-d*', '-d*', '-rn', 'dm', 'rn', 'd*', '-rn', '-dm', 'rn'], (1, 3, 2, 0): ['-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm', '-d*', '-d*', '-rm', 'db', 'rm', 'd*', '-rm', '-db', 'rm'], (3, 1, 0, 2): ['-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm', 'd*', 'd*', '-rm', '-dn', 'rm', '-d*', '-rm', 'dn', 'rm'], (0, 3, 1, 2): ['-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn', 'd*', 'd*', '-rn', '-dv', 'rn', '-d*', '-rn', 'dv', 'rn'], (2, 0, 1, 3): ['-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn', 'd*', 'd*', '-rn', '-dm', 'rn', '-d*', '-rn', 'dm', 'rn'], (3, 0, 2, 1): ['-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm', 'd*', 'd*', '-rm', '-db', 'rm', '-d*', '-rm', 'db', 'rm']}, {(2, 0, 1, 3): ['-dm', 'fn', 'dm', 'f0', '-dm', '-fn', 'dm', '-f0', '-f0', '-dm', 'fn', 'dm', 'f0', '-dm', '-fn', 'dm'], (0, 3, 1, 2): ['-dn', 'fv', 'dn', 'f0', '-dn', '-fv', 'dn', '-f0', '-f0', '-dn', 'fv', 'dn', 'f0', '-dn', '-fv', 'dn'], (3, 1, 0, 2): ['-dn', 'fm', 'dn', 'f0', '-dn', '-fm', 'dn', '-f0', '-f0', '-dn', 'fm', 'dn', 'f0', '-dn', '-fm', 'dn'], (3, 0, 2, 1): ['-dm', 'fb', 'dm', 'f0', '-dm', '-fb', 'dm', '-f0', '-f0', '-dm', 'fb', 'dm', 'f0', '-dm', '-fb', 'dm'], (1, 2, 0, 3): ['-dm', '-fn', 'dm', '-f0', '-dm', 'fn', 'dm', 'f0', 'f0', '-dm', '-fn', 'dm', '-f0', '-dm', 'fn', 'dm'], (0, 2, 3, 1): ['-dn', '-fv', 'dn', '-f0', '-dn', 'fv', 'dn', 'f0', 'f0', '-dn', '-fv', 'dn', '-f0', '-dn', 'fv', 'dn'], (2, 1, 3, 0): ['-dn', '-fm', 'dn', '-f0', '-dn', 'fm', 'dn', 'f0', 'f0', '-dn', '-fm', 'dn', '-f0', '-dn', 'fm', 'dn'], (1, 3, 2, 0): ['-dm', '-fb', 'dm', '-f0', '-dm', 'fb', 'dm', 'f0', 'f0', '-dm', '-fb', 'dm', '-f0', '-dm', 'fb', 'dm']}, {(3, 1, 0, 2): ['-dn', 'rm', 'dn', 'r0', '-dn', '-rm', 'dn', '-r0', '-r0', '-dn', 'rm', 'dn', 'r0', '-dn', '-rm', 'dn'], (3, 0, 2, 1): ['-dm', 'rb', 'dm', 'r0', '-dm', '-rb', 'dm', '-r0', '-r0', '-dm', 'rb', 'dm', 'r0', '-dm', '-rb', 'dm'], (2, 0, 1, 3): ['-dm', 'rn', 'dm', 'r0', '-dm', '-rn', 'dm', '-r0', '-r0', '-dm', 'rn', 'dm', 'r0', '-dm', '-rn', 'dm'], (0, 3, 1, 2): ['-dn', 'rv', 'dn', 'r0', '-dn', '-rv', 'dn', '-r0', '-r0', '-dn', 'rv', 'dn', 'r0', '-dn', '-rv', 'dn'], (2, 1, 3, 0): ['-dn', '-rm', 'dn', '-r0', '-dn', 'rm', 'dn', 'r0', 'r0', '-dn', '-rm', 'dn', '-r0', '-dn', 'rm', 'dn'], (1, 3, 2, 0): ['-dm', '-rb', 'dm', '-r0', '-dm', 'rb', 'dm', 'r0', 'r0', '-dm', '-rb', 'dm', '-r0', '-dm', 'rb', 'dm'], (1, 2, 0, 3): ['-dm', '-rn', 'dm', '-r0', '-dm', 'rn', 'dm', 'r0', 'r0', '-dm', '-rn', 'dm', '-r0', '-dm', 'rn', 'dm'], (0, 2, 3, 1): ['-dn', '-rv', 'dn', '-r0', '-dn', 'rv', 'dn', 'r0', 'r0', '-dn', '-rv', 'dn', '-r0', '-dn', 'rv', 'dn']}, {(1, 3, 2, 0): ['-dm', 'fn', 'dm', 'f*', '-dm', '-fn', 'dm', '-f*', '-f*', '-dm', 'fn', 'dm', 'f*', '-dm', '-fn', 'dm'], (2, 1, 3, 0): ['-dn', 'fv', 'dn', 'f*', '-dn', '-fv', 'dn', '-f*', '-f*', '-dn', 'fv', 'dn', 'f*', '-dn', '-fv', 'dn'], (0, 2, 3, 1): ['-dn', 'fm', 'dn', 'f*', '-dn', '-fm', 'dn', '-f*', '-f*', '-dn', 'fm', 'dn', 'f*', '-dn', '-fm', 'dn'], (1, 2, 0, 3): ['-dm', 'fb', 'dm', 'f*', '-dm', '-fb', 'dm', '-f*', '-f*', '-dm', 'fb', 'dm', 'f*', '-dm', '-fb', 'dm'], (3, 0, 2, 1): ['-dm', '-fn', 'dm', '-f*', '-dm', 'fn', 'dm', 'f*', 'f*', '-dm', '-fn', 'dm', '-f*', '-dm', 'fn', 'dm'], (3, 1, 0, 2): ['-dn', '-fv', 'dn', '-f*', '-dn', 'fv', 'dn', 'f*', 'f*', '-dn', '-fv', 'dn', '-f*', '-dn', 'fv', 'dn'], (0, 3, 1, 2): ['-dn', '-fm', 'dn', '-f*', '-dn', 'fm', 'dn', 'f*', 'f*', '-dn', '-fm', 'dn', '-f*', '-dn', 'fm', 'dn'], (2, 0, 1, 3): ['-dm', '-fb', 'dm', '-f*', '-dm', 'fb', 'dm', 'f*', 'f*', '-dm', '-fb', 'dm', '-f*', '-dm', 'fb', 'dm']}, {(0, 2, 3, 1): ['-dn', 'rm', 'dn', 'r*', '-dn', '-rm', 'dn', '-r*', '-r*', '-dn', 'rm', 'dn', 'r*', '-dn', '-rm', 'dn'], (1, 2, 0, 3): ['-dm', 'rb', 'dm', 'r*', '-dm', '-rb', 'dm', '-r*', '-r*', '-dm', 'rb', 'dm', 'r*', '-dm', '-rb', 'dm'], (1, 3, 2, 0): ['-dm', 'rn', 'dm', 'r*', '-dm', '-rn', 'dm', '-r*', '-r*', '-dm', 'rn', 'dm', 'r*', '-dm', '-rn', 'dm'], (2, 1, 3, 0): ['-dn', 'rv', 'dn', 'r*', '-dn', '-rv', 'dn', '-r*', '-r*', '-dn', 'rv', 'dn', 'r*', '-dn', '-rv', 'dn'], (0, 3, 1, 2): ['-dn', '-rm', 'dn', '-r*', '-dn', 'rm', 'dn', 'r*', 'r*', '-dn', '-rm', 'dn', '-r*', '-dn', 'rm', 'dn'], (2, 0, 1, 3): ['-dm', '-rb', 'dm', '-r*', '-dm', 'rb', 'dm', 'r*', 'r*', '-dm', '-rb', 'dm', '-r*', '-dm', 'rb', 'dm'], (3, 0, 2, 1): ['-dm', '-rn', 'dm', '-r*', '-dm', 'rn', 'dm', 'r*', 'r*', '-dm', '-rn', 'dm', '-r*', '-dm', 'rn', 'dm'], (3, 1, 0, 2): ['-dn', '-rv', 'dn', '-r*', '-dn', 'rv', 'dn', 'r*', 'r*', '-dn', '-rv', 'dn', '-r*', '-dn', 'rv', 'dn']}, {(3, 0, 2, 1): ['-rm', 'dn', 'rm', 'd0', '-rm', '-dn', 'rm', '-d0', '-d0', '-rm', 'dn', 'rm', 'd0', '-rm', '-dn', 'rm'], (2, 0, 1, 3): ['-rn', 'dv', 'rn', 'd0', '-rn', '-dv', 'rn', '-d0', '-d0', '-rn', 'dv', 'rn', 'd0', '-rn', '-dv', 'rn'], (0, 3, 1, 2): ['-rn', 'dm', 'rn', 'd0', '-rn', '-dm', 'rn', '-d0', '-d0', '-rn', 'dm', 'rn', 'd0', '-rn', '-dm', 'rn'], (3, 1, 0, 2): ['-rm', 'db', 'rm', 'd0', '-rm', '-db', 'rm', '-d0', '-d0', '-rm', 'db', 'rm', 'd0', '-rm', '-db', 'rm'], (1, 3, 2, 0): ['-rm', '-dn', 'rm', '-d0', '-rm', 'dn', 'rm', 'd0', 'd0', '-rm', '-dn', 'rm', '-d0', '-rm', 'dn', 'rm'], (1, 2, 0, 3): ['-rn', '-dv', 'rn', '-d0', '-rn', 'dv', 'rn', 'd0', 'd0', '-rn', '-dv', 'rn', '-d0', '-rn', 'dv', 'rn'], (0, 2, 3, 1): ['-rn', '-dm', 'rn', '-d0', '-rn', 'dm', 'rn', 'd0', 'd0', '-rn', '-dm', 'rn', '-d0', '-rn', 'dm', 'rn'], (2, 1, 3, 0): ['-rm', '-db', 'rm', '-d0', '-rm', 'db', 'rm', 'd0', 'd0', '-rm', '-db', 'rm', '-d0', '-rm', 'db', 'rm']}]    



surf_4perm = {
    (0, 1, 2, 3): [], 
    (3, 1, 0, 2): [(2, 1, 3, 0)], 
    (2, 0, 1, 3): [(1, 2, 0, 3)], 
    (0, 3, 1, 2): [(0, 2, 3, 1)], 
    (3, 0, 2, 1): [(1, 3, 2, 0)], 
    (2, 1, 3, 0): [(3, 1, 0, 2)], 
    (1, 2, 0, 3): [(2, 0, 1, 3)], 
    (0, 2, 3, 1): [(0, 3, 1, 2)], 
    (1, 3, 2, 0): [(3, 0, 2, 1)], 
    (2, 3, 0, 1): [(2, 1, 3, 0), (0, 2, 3, 1)], 
    (1, 0, 3, 2): [(2, 1, 3, 0), (1, 3, 2, 0)], 
    (3, 2, 1, 0): [(2, 1, 3, 0), (2, 0, 1, 3)]
}


def translate_move_surf(moves, cube_size, x, y):
    out = []
    for move in moves:
        if move == "":
            continue
        m = move.replace("*", str(cube_size-1))
        m = m.replace("v", str(x))
        m = m.replace("b", str(cube_size - 1 - x))
        m = m.replace("n", str(y))
        m = m.replace("m", str(cube_size - 1 - y))
        if "x" in move:
            # doubleの処理
            out.append(m.replace("x", ""))
            out.append(m.replace("x", ""))
        elif "^" in move:
            for i in range(cube_size):
                out.append(m.replace("^", str(i)))
        elif "i" in move:
            out.append(move.replace("i", "0"))
            out.append(move.replace("i", str(cube_size-1)))
        else:
            out.append(m)

    return out


def get_4_surface(state, final, cube_size, idx, x, y, force_final=False):
    if not force_final:
        tgts = get_target_for_final(state, final, cube_size, idx)
    state_indice = [
        cube_size*cube_size*idx + x + y * cube_size,
        cube_size*cube_size*idx + (cube_size- 1-y) + x * cube_size,
        cube_size*cube_size*idx + (cube_size- 1-x) + (cube_size- 1-y) * cube_size,
        cube_size*cube_size*idx + y + (cube_size- 1-x) * cube_size,
    ]

    tgt_indice = [
        x-1 + (y-1) * (cube_size-2),
        (cube_size-2-y) + (x -1) * (cube_size-2),
        (cube_size-2-x) + (cube_size-2-y) * (cube_size-2),
        (y-1) + (cube_size-2-x) * (cube_size-2),
    ]

    out_alias = [-1, -1, -1, -1]


    for key in surf_4perm:
        isok = True
        for i in range(4):
            orig = state[state_indice[i]]
            if force_final:
                tgt = final[state_indice[key[i]]]
            else:
                tgt = tgts[tgt_indice[key[i]]]
            if orig != tgt:
                isok = False
                break         
        if isok:
            return key
    # illeagalなpermは出ないことを前提
    for i in range(4):
        orig = state[state_indice[i]]
        if force_final:
            tgt = final[state_indice[key[i]]]
        else:
            tgt = tgts[tgt_indice[key[i]]]
        print(orig, tgt)
    raise ValueError

    """
    # aliasはこっち向きでいいの？
    for i in range(4):
        orig = state[state_indice[i]]
        #print(orig)
        for j in range(4):
            if j in out_alias:
                continue
            if force_final:
                tgt = final[state_indice[j]]
            else:
                tgt = tgts[tgt_indice[j]]            
            if tgt == orig:
                out_alias[i] = j
                break
    """
    return tuple(out_alias)

    
def get_target_for_final(state, final, cube_size, idx):
    # 面の中央ごとにゴールとなる面の情報を取得
    center_dict = get_center_dict(final)
    # 現状のstateの指定したidxのcenterを取り出す
    key = get_center(state, cube_size, idx)
    # centerの情報をkeyにその面が満たすべきゴールを取得する
    center_char = center_dict[key]

    idx_alias_0 = [i for i in range((cube_size-2)**2)]
    idx_alias_1 = [(cube_size-2)**2 - 1 - i for i in range((cube_size-2)**2)]
    idx_alias_2 = [ (cube_size - 2) ** 2 - (cube_size - 2) + (i // (cube_size - 2)) - (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]
    idx_alias_3 = [ (cube_size - 2) - 1 - (i // (cube_size - 2)) + (i % (cube_size - 2)) * (cube_size - 2) for i in range((cube_size-2)**2)]

    best_tgt = None
    best_v = 111111

    for alias in [idx_alias_0, idx_alias_1, idx_alias_2, idx_alias_3]:
        tgts = [center_char[idx] for idx in alias]
        vdic = {}

        dead = False
        v = 0
        for x_in in range(cube_size // 2):
            for y_in in range(cube_size // 2 - 1):
                x = x_in + 1
                y = y_in + 1
                #print(tgts)
                state_indice = [
                    cube_size*cube_size*idx + x + y * cube_size,
                    cube_size*cube_size*idx + (cube_size- 1-y) + x * cube_size,
                    cube_size*cube_size*idx + (cube_size- 1-x) + (cube_size- 1-y) * cube_size,
                    cube_size*cube_size*idx + y + (cube_size- 1-x) * cube_size,
                ]
                tgt_indice = [
                    x-1 + (y-1) * (cube_size-2),
                    (cube_size-2-y) + (x -1) * (cube_size-2),
                    (cube_size-2-x) + (cube_size-2-y) * (cube_size-2),
                    (y-1) + (cube_size-2-x) * (cube_size-2),
                ]

                out_alias = [-1, -1, -1, -1]

                # aliasはこっち向きでいいの？
                for i in range(4):
                    orig = state[state_indice[i]]
                    #print(orig)
                    for j in range(4):
                        tgt = tgts[tgt_indice[j]]            
                        #print(tgt)
                        if tgt == orig:
                            out_alias[i] = j
                            break

                cost = tuple(out_alias)
                if cost not in surf_4perm:
                    print("dead")
                    dead = True
                    break
                # print(cost)
                if cost in vdic:
                    vdic[cost] += 1
                else:
                    vdic[cost] = 1
                v += len(surf_4perm[cost])
            if dead:
                break
        if not dead:
            print(vdic)
            if best_v > v:
                best_v = v
                best_tgt = tgts
        
    print("bestv", best_v)
    return best_tgt        

def solve_final_surface(state, final_state, cube_size, idx, legal_moves, perm, force_final=False):
    total_moves = []
    score = 0
    if cube_size % 2 == 0:
        smax1 = cube_size // 2
        smax2 = cube_size // 2
    else:
        smax1 = cube_size // 2
        smax2 = smax1 - 1
    for i in range(smax1):
        for j in range(smax2):
            # print(get_target_for_final(state, final_state, 33, 0))
            ptu = get_4_surface(state, final_state, cube_size, idx, i+1, j+1, force_final)
            print(ptu, "before", i, j)
            mcs = surf_4perm[ptu]
            moves = []
            for mc in mcs:
                if i != j:
                    moves.extend(get_rev(surf_perm_xy[idx][mc]))
                else:
                    moves.extend(get_rev(surf_perm_xx[idx][mc]))
            moves = translate_move_surf(moves, cube_size, i+1, j+1)
            total_moves.extend(moves)
            state, perm = run_moves(state, moves, legal_moves, perm)
            ptu = get_4_surface(state, final_state, cube_size, idx, i+1, j+1, force_final)
            print(ptu, "after", i, j)
            score += len(moves)
    
    return state, total_moves, perm

### 最後の面の絵柄を揃えるソルバここまで ###


### 内側の面の回転 ###

rp_dict_all = {(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): [], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['d*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['d*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['d*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 12, 13, 14, 15, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 16, 17, 18, 19, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 4, 5, 6, 7, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 4, 5, 6, 7, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 5, 7, 4, 6, 8, 9, 10, 11, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 6, 4, 7, 5, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r0.-f*.f0.-r0.-r0.f*.-f0.-r0.-f*.f0.-r0.-r0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (2, 0, 3, 1, 5, 7, 4, 6, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.-d0.ri.-fi.di.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 6, 4, 7, 5, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.fi.-ri.d0.ri.-fi.di.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.fi.-ri.-d*.ri.-fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 22, 20, 23, 21): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (0, 1, 2, 3, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (0, 1, 2, 3, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-r*.-f*.f0.-r*.-r*.f*.-f0.-r*.-f*.f0.-r*.-r*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 21, 23, 20, 22): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*', '-di.ri.fi.d*.-fi.-ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 8, 9, 10, 11, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (2, 0, 3, 1, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 8, 9, 10, 11, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 20, 21, 22, 23): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.-d0.-fi.-ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.ri.fi.-d0.-fi.-ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.ri.fi.d0.-fi.-ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 16, 17, 18, 19, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 17, 19, 16, 18, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.f*.-ri.-di.fi.r0', 'f*.-f0.-d0.-f*.f0.-d0.-d0.f*.-f0.-d0.-f*.f0.-d0.-d0'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.-ri.-fi.d0.fi.ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 18, 16, 19, 17, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', 'f*.-f0.-d*.-f*.f0.-d*.-d*.f*.-f0.-d*.-f*.f0.-d*.-d*'], (3, 2, 1, 0, 5, 7, 4, 6, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d*.fi.ri.di.-f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (1, 3, 0, 2, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f*.-ri.-di.fi.r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d*.-fi.-ri.di.f0', '-di.ri.fi.d0.-fi.-ri.di.f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.-d0.ri.-fi.di.r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.fi.-ri.d*.ri.-fi.di.-r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.di.ri.-f0.-ri.-di.fi.-r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.d*.fi.ri.di.-f*'], (2, 0, 3, 1, 6, 4, 7, 5, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 5, 7, 4, 6, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.d0.ri.-fi.di.-r0', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.fi.-ri.-d*.ri.-fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 13, 15, 12, 14, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 22, 20, 23, 21): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.di.ri.f0.-ri.-di.fi.r*', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 5, 7, 4, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.di.ri.-f*.-ri.-di.fi.-r0', '-di.-ri.-fi.d0.fi.ri.di.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 14, 12, 15, 13, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0'], (3, 2, 1, 0, 6, 4, 7, 5, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.di.ri.f0.-ri.-di.fi.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.ri.fi.d0.-fi.-ri.di.f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 17, 19, 16, 18, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.d0.fi.ri.di.-f0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.-ri.-fi.-d0.fi.ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-di.-fi.ri.-d0.-ri.fi.di.-r*', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (2, 0, 3, 1, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 21, 23, 20, 22): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-fi.-di.-ri.-f*.ri.di.fi.r*', '-di.-ri.-fi.-d0.fi.ri.di.f0'], (3, 2, 1, 0, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 18, 16, 19, 17, 23, 22, 21, 20): ['-di.-fi.ri.-d*.-ri.fi.di.-r0', '-fi.-di.-ri.f*.ri.di.fi.-r*', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d0.fi.ri.di.f0', '-di.-ri.-fi.-d*.fi.ri.di.f*'], (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-fi.-di.-ri.-f0.ri.di.fi.r0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (2, 0, 3, 1, 7, 6, 5, 4, 10, 8, 11, 9, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.-f0.ri.di.fi.r0', '-di.ri.fi.-d*.-fi.-ri.di.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (1, 3, 0, 2, 7, 6, 5, 4, 9, 11, 8, 10, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-fi.-di.-ri.f0.ri.di.fi.-r0', '-di.ri.fi.d*.-fi.-ri.di.f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*'], (3, 2, 1, 0, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-ri.-fi.-d*.fi.ri.di.f*', '-di.-ri.-fi.-d*.fi.ri.di.f*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0'], (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20): ['-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d*.-ri.fi.di.r0', '-di.-fi.ri.d0.-ri.fi.di.r*', '-di.-fi.ri.d0.-ri.fi.di.r*', 'd*.-d0.-f0.-d*.d0.-f0.-f0.d*.-d0.-f0.-d*.d0.-f0.-f0', 'd*.-d0.-f*.-d*.d0.-f*.-f*.d*.-d0.-f*.-d*.d0.-f*.-f*']}

def get_surf_perm(state, final, cube_size):
    
    state_24 = []
    final_24 = []
    for i in range(6):
        bias = cube_size ** 2 * i
        state_24.append(bias + cube_size + 1)
        state_24.append(bias + cube_size*2-2)
        state_24.append(bias + cube_size * (cube_size - 2) + 1 )
        state_24.append(bias + cube_size * (cube_size-1)-2)
        
    int_24 = [state[s] for s in state_24]
    final_24 = [final[s] for s in state_24]

    print(int_24)
    print(final_24)

    for key in rp_dict_all:
        isok = True
        for i in range(24):
            orig = int_24[i]
            tgt = final_24[key[i]]
            if orig != tgt:
                isok = False
                break
        if isok:
            return key
    
    key = (0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8, 12, 13, 14, 15, 19, 18, 17, 16, 23, 22, 21, 20)
    for i in range(24):
        orig = int_24[i]
        tgt = final_24[key[i]]
        print(orig, tgt)
        if orig != tgt:
            isok = False
            break
    if isok:
        return key
    
    raise ValueError


def solve_surface_rotation(state, cube_size, perm, legal_moves, final):
    sperm = get_surf_perm(state, final, cube_size)
    # print(sperm)

    moves = ".".join(rp_dict_all[tuple(sperm)]).split(".")

    state, perm = run_moves(state, get_rev(translate_move_surf(moves, cube_size, 0, 0)), legal_moves, perm)    

    print(len(get_rev(translate_move_surf(moves, cube_size, 0, 0))), "moves")

    return state, get_rev(translate_move_surf(moves, cube_size, 0, 0)), perm


### 内側の面の回転ここまで ###



def solve_edge(state, cube_size, legal_moves, perm):
    mvall = []
    for _ in range(333):
        if count_complete_edge(state, cube_size) >= 10:
            break
        state, mvs = complete_edge(state, cube_size, 8, legal_moves, perm)
        mvall.extend(mvs)
        
        for j in range(8):
            if not is_edge_complete(state, cube_size, j):
                swap_moves = [m for m in swap_sac_moves[f"8-{j}"]]
                swap_moves = translate_move(swap_moves, cube_size)
                mvall.extend(swap_moves)
                state, perm = run_moves(state, swap_moves, legal_moves, perm)
                print(count_complete_edge(state, cube_size), "edgedone")
                print("swap", j)
                print_edge(state, cube_size)
                print("swapdone")
                break

    for paritr in range(111111):
        state, pmoves, pmlen, tmlen = remove_parity(state ,cube_size, 8, 9, legal_moves, perm)
        mvall.extend(pmoves)
        if is_edge_complete(state, cube_size, 8):
            break

    return state, mvall, perm, pmlen, tmlen



def solve_center_fast(state, cube_size, legal_moves, perm, final=None):
    debug_surf = False
    
    total_moves = []
    # 最もmaskがでかい手を探す
    cmoves = []
    for citr in range(6):
        max_score = 0
        if final is not None:
            to_candidate = citr
            tgt = None
        else:
            for candidate in range(6):
                tgt = get_center(state, cube_size, candidate)
                score = count_char(state, cube_size, candidate, tgt)
                if score > max_score and score < (cube_size -2) * (cube_size-2):
                    max_score = score
                    to_candidate = candidate
            tgt = get_center(state, cube_size, to_candidate)

        if final is not None and to_candidate == 0:
            continue

        for frm_candidate in range(6):
            #if final is not None and frm_candidate <= to_candidate and to_candidate != 5:
            #    continue
            if frm_candidate == to_candidate:
                continue
            
            for trial in range(111111):
                best_from = -1
                best_premove = []
                best_mask = []
                best_mask_size = 0

                for preswap in [[], [surf_counter_clock[to_candidate]], [get_rev(surf_counter_clock[to_candidate])], [surf_counter_clock[to_candidate], surf_counter_clock[to_candidate]]]:
                    for preswap2 in [[], [surf_counter_clock[frm_candidate]], [get_rev(surf_counter_clock[frm_candidate])], [surf_counter_clock[frm_candidate], surf_counter_clock[frm_candidate]]]:
                        temp_state = state
                        temp_perm = [p for p in perm]
                        premove = preswap + preswap2
                        temp_state, temp_perm = run_moves(temp_state, translate_move(premove, cube_size), legal_moves, temp_perm)
                        if tgt is not None:
                            mask = get_center_swap_mask(temp_state, cube_size, to_candidate, frm_candidate, tgt)
                        else:
                            mask = get_center_swap_mask_with_img(temp_state, cube_size, to_candidate, frm_candidate, final)
                        if len(mask) == 0:    
                            continue
                        mask_to_use = find_max_subset(mask)

                        if best_mask_size < len(mask_to_use):
                            best_mask_size = len(mask_to_use)
                            best_mask = mask_to_use
                            best_from = frm_candidate
                            best_to = to_candidate
                            best_premove = translate_move(premove, cube_size)

                # print best
                if tgt is None and debug_surf:            
                    temp_state = state
                    temp_perm = [p for p in perm]
                    temp_state, temp_perm = run_moves(temp_state, best_premove, legal_moves, temp_perm)
                    mask = get_center_swap_mask_with_img(temp_state, cube_size, to_candidate, frm_candidate, final, print_mask=True)

                print(best_mask_size, frm_candidate, to_candidate, best_mask)
                if best_mask_size == 0:
                    break
                    
                state, perm = run_moves(state, best_premove, legal_moves, perm)
                total_moves.extend(best_premove)
                moves, fscore = swap_to_surface(
                    best_to, 
                    best_from,
                    best_mask,
                    cube_size
                )

                # debug
                if debug_surf:
                    print_surface(state, cube_size, best_from)

                    print("---------------")
                    print_surface(state, cube_size, best_to)
                    print("****************")
                
                moves = translate_move(moves, cube_size)
                total_moves.extend(moves)
                state, perm = run_moves(state, moves, legal_moves, perm)
                
                if debug_surf:
                    print_surface(state, cube_size, best_from)
                    print("---------------")
                    print_surface(state, cube_size, best_to)
                    print("****************")

                    _ = input()
        
        cm = len(total_moves)
        for c in cmoves:
            cm -= c
        cmoves.append(cm)
        

    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")

    
    print(len(total_moves), "to centerize", cmoves)
    # raise ValueError
    return state, total_moves


class BaseEvaluator:
    def __init__(self, type:str, initial_state:str, final_state:str, wildcard:int) -> None:
        self.type = type
        self.initial_state = initial_state
        self.final_state = final_state
        self.wildcard = wildcard

    def evaluate(self, state:List[str]) -> int:
        """
        スコア計算。クリアした時を0に固定したいので不一致なものを減点していく形式に
        """
        score = 0
        for f, s in zip(self.final_state, state):
            if f != s:
                score += 1
        return score

    def completed(self, state:List[str]) -> bool:
        # 問題によっては特殊な処理を詰める
        value = self.evaluate(state)
        if value <= self.wildcard:
            return True
        return False
    

class EvaluatorCubeCenter(BaseEvaluator):
    def __init__(self, type: str, initial_state: str, final_state: str, wildcard: int) -> None:
        super().__init__(type, initial_state, final_state, wildcard)
        assert "cube" in type
        self.cube_size = int(self.type.split("/")[-1])
        self.center_idx = []
        self.cc_idx = []
        for i in range(6):
            bias = self.cube_size * self.cube_size * i
            idx = []
            for j in range(self.cube_size-2):
                for k in range(self.cube_size-2):
                    idx.append(bias + self.cube_size * (j+1) + 1 + k)
            self.center_idx.append(idx)
            self.cc_idx.append(idx[(len(idx)+1)//2-1])

        self.wildcard = 0
        #print(self.center_idx)
        #print(self.cc_idx)

        
    def evaluate(self, state: List[str]) -> int:
        score = 0
        for i in range(6):
            # score += (self.cube_size-2) * (self.cube_size-2)
            for idx in self.center_idx[i]:
                if state[idx] == state[self.cc_idx[i]]:
                    score += 1
        return score

class FinalState:
    def __init__(self, final_state:List[str], moves:List[str]):
        self.final_state = final_state
        self.moves = moves


def get_shortest_path(
    moves: dict[str, tuple[int, ...]], max_size: int, max_depth=100) -> dict[tuple[int, ...], list[str]]:
    n = len(next(iter(moves.values())))

    state = tuple(range(n))
    cur_states = [state]

    shortest_path: dict[tuple[int, ...], list[str]] = {}
    shortest_path[state] = []

    for _ in range(max_depth):
        next_states = []
        for state in cur_states:
            for move_name, perm in moves.items():
                next_state = tuple(state[i] for i in perm)
                if next_state in shortest_path:
                    continue
                shortest_path[next_state] = shortest_path[state] + [move_name]
                next_states.append(next_state)
                if len(shortest_path) > max_size:
                    return shortest_path
        cur_states = next_states
    return shortest_path


class BaseSolver:

    def __init__(self, evaluator:BaseEvaluator, one_find_better=False) -> None:
        self.evaluator = evaluator
        self.one_find_better = one_find_better

    def generate_path(self, legal_moves:dict[str, tuple[int, ...]], max_size,max_depth) -> dict[tuple[int, ...], list[str]]:
        self.spath_dict = get_shortest_path(legal_moves, max_size=max_size,max_depth=max_depth)

    def search_next(self, initial_state:List[str]) -> FinalState:
        initial_value = self.evaluator.evaluate(initial_state)
        best_value = initial_value
        best_state = initial_state
        best_path = []

        for path in self.spath_dict:
            state = [initial_state[i] for i in path]
            value = self.evaluator.evaluate(state)
            if value < best_value:
                best_state = state
                best_value = value
                best_path = self.spath_dict[path]
                if self.one_find_better:
                    break
        print(initial_value, "to", best_value, "with", len(best_path))
        return FinalState(best_state, best_path)


def add_rot_suff(cubesize):
    out = [""]
    for mv in ["f", "-f", "r", "-r", "d", "-d"]:
        tmp = ""
        for i in range(cubesize):
            tmp += "." + mv + str(i)
        out.append(tmp)

    out2 = []
    for o in out:
        for mv in ["f", "-f", "r", "-r", "d", "-d"]:
            tmp = ""
            for i in range(cubesize):
                tmp += "." + mv + str(i)
            out2.append(o+tmp)
            
    out3 = []
    for o in out2:
        for mv in ["f", "-f", "r", "-r", "d", "-d"]:
            tmp = ""
            for i in range(cubesize):
                tmp += "." + mv + str(i)
            out3.append(o+tmp)
    out.extend(out2)
    out.extend(out3)
    return out

def get_best_orient(state, final_state, legal_moves, cube_size):
    suffs = add_rot_suff(cube_size)
    max_score = -1
    best_suff = []
    for suff in suffs:
        temp_state = [s for s in state]
        if suff != "":
            temp_state, _ = run_moves(temp_state, suff.split(".")[1:], legal_moves, [i for i in range(len(state))])
        score = 0
        idx = 0
        for s, f in zip(temp_state, final_state):
            idx += 1
            if idx % cube_size ** 2 != 1:
                continue
            if s == f:
                score += 1
        # print(score, "orient score")
        if score > max_score:
            max_score = score
            best_suff = suff.split(".")[1:]
        if score == len(state):
            print("solved!")
            break
    return best_suff


def solve(puzzles_path:str="data/puzzles.csv", sln_path:str="data/merged_122017.csv", puzzle_info_path:str="data/puzzle_info.csv", solver_path_444="data/merged_122017.csv", solver_path:str="/home/shiku/AI/kaggle/santa2023/rubiks-cube-NxNxN-solver", id:int=281, initial_state:str=None, initial_moves:str=None, output_path:str="sln.txt", use_anl:bool=False, skip_alias:bool=False, force_skip_surf:bool=False):
    import pandas as pd
    tilewise = False
    puzzle = pd.read_csv(puzzles_path).set_index("id").loc[id]
    sln = pd.read_csv(sln_path).set_index("id").loc[id]
    sln_moves = sln["moves"].split(".")
    if initial_state is None:
        initial_state = puzzle["initial_state"].split(";")
    else:
        initial_state = initial_state.split(";")
    perm = [i for i in range(len(initial_state))]
    final_state = puzzle["solution_state"].split(";")
    cube_size = int(puzzle["puzzle_type"].split("/")[-1])
    wildcard = int(puzzle["num_wildcards"])
    legal_moves = get_moves(puzzle["puzzle_type"], puzzle_info_path)
    alias = None
    if not skip_alias:
        alias = get_alias(initial_state, sln_moves, legal_moves)
    #print(alias)
    state = initial_state if alias is None else [a for a in alias]
    center_dict = get_center_dict(final_state)
    #print(center_dict)
    movelen = 0
    center_moves = []
    total_moves = []

    evaluator = EvaluatorCubeCenter(puzzle["puzzle_type"], initial_state=initial_state, final_state=final_state, wildcard=wildcard)
    solver = BaseSolver(evaluator=evaluator)
    solver.generate_path(legal_moves, 111111, 1)

    value_before = evaluator.evaluate(state)
    value_after = -1

    if initial_moves is not None:
        score = 0
        for s, f in zip(state, final_state):
            if s == f:
                score += 1
        print(score, len(state))

        imoves = initial_moves.split(".")
        total_moves.extend(imoves)
        intmove = len(imoves)
        state, perm = run_moves(state, imoves, legal_moves, perm)
        value_after = evaluator.evaluate(state)

        score = 0
        for s, f in zip(state, final_state):
            if s == f:
                score += 1
        print(score, len(state))
        print(value_after, value_before)
        # raise ValueError


    else:
        intmove = 0

    anl = 0

    if use_anl:
        for _ in range(10000):
            out = solver.search_next(initial_state=state)
            state = out.final_state
            total_moves.extend(out.moves)
            anl += len(out.moves)
            if len(out.moves) == 0:
                break
        print(anl, "for anneal")
    
    

    if not force_skip_surf:
        for itr in range(6):
            to_max = 0
            max_score = 0
            for candidate in range(6):
                center = get_center(state, cube_size, candidate)
                tgt = center_dict[center] if tilewise else center
                score = count_char(state, cube_size, candidate, tgt)
                if score > max_score and score < (cube_size -2) * (cube_size-2):
                    max_score = score
                    to = candidate
            to = itr
            cmove = 0
            center = get_center(state, cube_size, to)
            tgt = center_dict[center] if tilewise else center
            score = count_char(state, cube_size, to, tgt)
            for fitr in range(6):
                frm = fitr
                if frm == to:
                    continue

                for i in range(111111):
                    score_before_frm = count_char(state, cube_size, frm, tgt)
                    if score_before_frm == 0:
                        print("nosurf")
                        # print_surface(state, cube_size, frm)
                        print("********")
                        break
                    mask = get_center_swap_mask(state, cube_size,to, frm, tgt)
                    if len(mask) == 0:
                        mmoves = translate_move([surf_counter_clock[to]],cube_size)
                        total_moves.extend(mmoves)
                        state, perm = run_moves(state, mmoves, legal_moves, perm)

                        movelen += 1
                        cmove += 1
                        # print_surface(state, cube_size, to)
                        continue
                    mask_to_use = find_max_subset(mask)    
                    moves, fscore = swap_to_surface(
                            to, 
                            frm,
                            mask_to_use,
                            cube_size
                    )

                    moves = translate_move(moves, cube_size)
                    total_moves.extend(moves)
                    #print(mask_to_use)
                    #print(moves)
                    #print_surface(state, cube_size, frm)
                    #print("==================")
                    #print_surface(state, cube_size, to)
                    #print("**************************")

                    
                    movelen += len(moves)
                    cmove += len(moves)
                    for mv in moves:
                        state, perm = run_moves(state, [mv], legal_moves, perm)
                        #print_surface(state, cube_size, to)
                        #print("**************************")
                    score = count_char(state, cube_size, to, tgt)
                    print(score_before_frm, score)
                    print("**************")
                    #if i == 5:
                    #    raise ValueError
            center_moves.append(cmove)
        print(movelen, "moves for center")
    else:
        print("force skip center")

    print_edge(state, cube_size)
    print("=============")
    mvall = []
    
    for _ in range(333):
        if count_complete_edge(state, cube_size) >= 10:
            break
        state, mvs = complete_edge(state, cube_size, 8, legal_moves, perm)
        mvall.extend(mvs)
        total_moves.extend(mvs)
        
        for j in range(8):
            if not is_edge_complete(state, cube_size, j):
                swap_moves = [m for m in swap_sac_moves[f"8-{j}"]]
                swap_moves = translate_move(swap_moves, cube_size)
                mvall.extend(swap_moves)
                total_moves.extend(swap_moves)
                state, perm = run_moves(state, swap_moves, legal_moves, perm)
                print(count_complete_edge(state, cube_size), "edgedone")
                print("swap", j)
                print_edge(state, cube_size)
                print("swapdone")
                break
    
    print(len(mvall), "moves for 10 edges")
    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")
    parmove = []
    for paritr in range(111111):
        state, pmoves, pmlen, tmlen = remove_parity(state ,cube_size, 8, 9, legal_moves, perm)
        total_moves.extend(pmoves)
        parmove.extend(pmoves)
        if is_edge_complete(state, cube_size, 8):
            break
        
    print("jobdone", len(total_moves))
    print("solve333")
    # moves333 = solve_by_solver(initial_state)

    """
    moves333 = solve_force_333(state, cube_size)
    moves333_2 = solve_by_solver(state, solver_path=solver_path)
    print(moves333, len(moves333))
    print(moves333_2, len(moves333_2))
    raise ValueError
    """

    try:
        if cube_size % 2 == 0:
            #pll = solve_oll(cube_size)
            #state, perm = run_moves(state, translate_move(pll, cube_size), legal_moves, perm)
            moves333 = solve_by_solver(state, solver_path_444=solver_path_444)
        else:
            moves333 = solve_by_solver(state, solver_path=solver_path)
            # moves333 = solve_force_333(state, cube_size)
    except Exception:
        moves333 = []
        print("solver error, skip solver")

    total_moves.extend(moves333)
    state, perm = run_moves(state, moves333, legal_moves, perm)

    print("change to original", len(total_moves))
    state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])

    suff = get_best_orient(state, final_state, legal_moves, cube_size)
    state, perm = run_moves(state, suff, legal_moves, perm)
    total_moves.extend(suff)

    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")

    print("validation", len(total_moves))
    test_final_state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    for i in range(6):
        print_surface(test_final_state, cube_size, i)
        print("------------------------")
    print("score", len(total_moves))
    print(intmove, "initial_move",value_before, "value_before", value_after, "value_after", anl, "anneal", movelen, "center", center_moves, "each center", len(mvall), "edge", len(parmove), "parity", pmlen, tmlen, "(parity move stat) ",len(moves333), "solve333")
    
    with open(output_path ,"w") as f:
        f.write(".".join(total_moves))
        print("saved", output_path)

    score = 0
    for s, f in zip(test_final_state, final_state):
        if s == f:
            score += 1

    print(score, len(state))

    if score != len(state):
        print("valiation failed rotation test")
        isok = False
        for smove in ["r0", "r*", "-r0", "-r*", "r0x", "r*x", "d0", "d*", "-d0", "-d*", "d0x", "d*x", "f0", "f*", "-f0", "-f*", "f0x", "f*x"]:
            scopy = [s for s in test_final_state]
            pcopy = [p for p in perm]
            add_rot = translate_move([smove], cube_size)
            scopy, pcopy = run_moves(scopy, add_rot, legal_moves, pcopy)

            score = 0
            for s, f in zip(scopy, final_state):
                if s == f:
                    score += 1
            print(score, len(state))
            if score == len(state):
                total_moves.extend(add_rot)
                isok = True
                print("rotation ok!")
                for i in range(6):
                    print_surface(scopy, cube_size, i)
                    print("------------------------")
                print(score, len(state))
                with open(output_path ,"w") as f:
                    f.write(".".join(total_moves))
                    print("saved", output_path)                
                break

        if not isok:
            raise ValueError("validation failed!!")

    return state, total_moves, perm
    

from itertools import product

def generate_all_outputs(allow_char):
    all_chars = set().union(*allow_char)
    all_outputs = []

    for combination in product(*allow_char):
        if set(combination) == all_chars:
            all_outputs.append(list(combination))

    return all_outputs


def get_333_alias(id, initial_state, sln_moves, legal_moves, cube_size, total_moves, final_state, alias=None, surf_rot_perm=None, output_path="sln.txt") -> List[str]:
    """
    solver向けのaliasの候補を複数出す、ソルバを呼ぶ、といった処理を一貫で行う。偶数のABABのたぐいは特に邪魔なので注意
    """
    if alias is None:
        alias = get_alias(initial_state, sln_moves, legal_moves)
    

    valid_moves = total_moves
    valid_state, _ = run_moves(initial_state, valid_moves, legal_moves, [i for i in range(len(alias))])

    for i in range(6):
        print_surface(valid_state, cube_size, i)
        print("------------------------")
    

    alias_state, _ = run_moves(alias, total_moves, legal_moves, [i for i in range(len(alias))])
    for i in range(6):
        print_surface(alias_state, cube_size, i)
        print("------------------------")
    
    # center以外の色は割り振り直す必要はないはず
    # centerは一個でもそれを含むやつは候補になりうるとみなす
    c_candidate = []
    for i in range(6):
        cset = set()
        for y in range(cube_size):
            if y == 0 or y == cube_size - 1:
                continue
            for x in range(cube_size):
                if x == 0 or x == cube_size - 1:
                    continue
                idx = i * cube_size ** 2 + y * cube_size + x
                cset.add(alias_state[idx])
        c_candidate.append(cset)

    all_results = generate_all_outputs(c_candidate)
    # print(all_results)
    # 色々用意するが恐らくmurai-slnだと ABCDEF固定。なのでコードは用意したが使わぬ
    all_results = ["A", "B", "C", "D", "E", "F"]

    alias_state = list(alias_state)

    for i in range(6):
        for y in range(cube_size):
            for x in range(cube_size):
                if y == 0 or y == cube_size - 1:
                    if x == 0 or x == cube_size - 1:
                        continue
                idx = i * cube_size ** 2 + y * cube_size + x
                alias_state[idx] = all_results[i]


    moves333 = solve_by_solver(alias_state, force3=False)
    # moves333 = solve_force_333(alias_state, cube_size)

    
    valid_moves = total_moves + moves333
    valid_state, _ = run_moves(initial_state, valid_moves, legal_moves, [i for i in range(len(alias))])

    for i in range(6):
        print_surface(valid_state, cube_size, i)
        print("------------------------")
    
    suff = get_best_orient(valid_state, final_state, legal_moves, cube_size)
    valid_moves = total_moves + moves333 + suff
    valid_state, perm = run_moves(initial_state, valid_moves, legal_moves, [i for i in range(len(alias))])
    
    surf_rot_perm = []
    """
    # manual opt for 256
    surf_rot_perm = [
        2, 0, 3, 1,
        7, 6, 5, 4,
        8, 9, 10, 11,
        15,14,13,12,
        17,19,16,18,
        20,21,22,23
    ]
    """

    for i in range(6):
        print_surface(valid_state, cube_size, i)
        print("------------------------")
    

    if len(surf_rot_perm) == 0:
        valid_state, sumoves, perm = solve_surface_rotation(valid_state, cube_size, perm, legal_moves, final_state)
    else:
        sumoves = get_rev(translate_move_surf(".".join(rp_dict_all[tuple(surf_rot_perm)]).split("."), cube_size, 0, 0))
    
    valid_moves = total_moves + moves333 + suff + sumoves
    valid_state, _ = run_moves(initial_state, valid_moves, legal_moves, [i for i in range(len(alias))])


    for i in range(6):
        print_surface(valid_state, cube_size, i)
        print("------------------------")

    fp_moves = []
    final_perms = []
    """
    final_perms = [
        [5, 2, 1, (2, 0, 1, 3)],
        [5, 1, 1, (0, 3, 1, 2)],
        [5, 2, 3, (0, 3, 1, 2)]
    ]
    """

    for fp in final_perms:
        fp_moves.extend(translate_move_surf(get_rev(surf_perm_xy[fp[0]][fp[3]]), cube_size, fp[1], fp[2]))

    valid_moves = total_moves + moves333 + suff + sumoves + fp_moves
    valid_state, _ = run_moves(initial_state, valid_moves, legal_moves, [i for i in range(len(alias))])


    for i in range(6):
        print_surface(valid_state, cube_size, i)
        print("------------------------")
    
    score = 0
    for i in range(len(final_state)):
        if final_state[i] == valid_state[i]:
            score += 1
    print(score, len(final_state))

    #with open(f"sln_{id}_ok.txt", "w") as f:
    #    f.write(".".join(valid_moves))

    with open(output_path, "w") as f:
        f.write(".".join(valid_moves))
    print(len(valid_moves))




def solve_only333(puzzles_path:str="data/puzzles.csv", sln_path:str="data/merged_122017.csv", puzzle_info_path:str="data/puzzle_info.csv", solver_path_444="data/merged_122017.csv", solver_path:str="/home/shiku/AI/kaggle/santa2023/rubiks-cube-NxNxN-solver", id:int=281, initial_state:str=None, initial_moves:str=None, output_path:str="sln.txt"):
    import pandas as pd
    tilewise = False
    puzzle = pd.read_csv(puzzles_path).set_index("id").loc[id]
    sln = pd.read_csv(sln_path).set_index("id").loc[id]
    sln_moves = sln["moves"].split(".")
    if initial_state is None:
        initial_state = puzzle["initial_state"].split(";")
    else:
        initial_state = initial_state.split(";")
    perm = [i for i in range(len(initial_state))]
    final_state = puzzle["solution_state"].split(";")
    cube_size = int(puzzle["puzzle_type"].split("/")[-1])
    wildcard = int(puzzle["num_wildcards"])
    legal_moves = get_moves(puzzle["puzzle_type"], puzzle_info_path)
    alias = get_alias(initial_state, sln_moves, legal_moves)
    #print(alias)
    state = [a for a in alias]

    total_moves = []
    if initial_moves is not None:
        imoves = initial_moves.split(".")
        total_moves.extend(imoves)
        intmove = len(imoves)
        state, perm = run_moves(state, imoves, legal_moves, perm)
    else:
        intmove = 0

    total_moves.extend(solve_by_solver(state, solver_path=solver_path, force_odd=True))
    state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    suff = get_best_orient(state, final_state, legal_moves, cube_size)
    state, perm = run_moves(state, suff, legal_moves, perm)
    total_moves.extend(suff)

    for i in range(6):
        state, moves, perm =  solve_final_surface(state, final_state, cube_size, i, legal_moves, perm, True)
        print(i, "done")
        total_moves.extend(moves)


    print("validation", len(total_moves))
    test_final_state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    for i in range(6):
        print_surface(test_final_state, cube_size, i)
        print("------------------------")
    with open(output_path ,"w") as f:
        f.write(".".join(total_moves))

    score = 0
    for s, f in zip(state, final_state):
        if s == f:
            score += 1
    print("score", len(total_moves))
    print(score, len(state))


    
def solve_with_img(puzzles_path:str="data/puzzles.csv", sln_path:str="data/merged_122017.csv", puzzle_info_path:str="data/puzzle_info.csv", solver_path_444="data/merged_122017.csv", solver_path:str="/home/shiku/AI/kaggle/santa2023/rubiks-cube-NxNxN-solver", id:int=281, initial_state:str=None, initial_moves:str=None, output_path:str="sln.txt"):
    import pandas as pd
    tilewise = False
    puzzle = pd.read_csv(puzzles_path).set_index("id").loc[id]
    sln = pd.read_csv(sln_path).set_index("id").loc[id]
    sln_moves = sln["moves"].split(".")
    if initial_state is None:
        initial_state = puzzle["initial_state"].split(";")
    else:
        initial_state = initial_state.split(";")
    perm = [i for i in range(len(initial_state))]
    final_state = puzzle["solution_state"].split(";")
    cube_size = int(puzzle["puzzle_type"].split("/")[-1])
    wildcard = int(puzzle["num_wildcards"])
    legal_moves = get_moves(puzzle["puzzle_type"], puzzle_info_path)
    alias = get_alias(initial_state, sln_moves, legal_moves)
    #print(alias)
    state = [a for a in alias]

    total_moves = []

    if initial_moves is not None:
        imoves = initial_moves.split(".")
        total_moves.extend(imoves)
        intmove = len(imoves)
        state, perm = run_moves(state, imoves, legal_moves, perm)
    else:
        intmove = 0

    state, edgemoves, perm, pmlen, tmlen = solve_edge(state, cube_size, legal_moves, perm)
    total_moves.extend(edgemoves)

    # edgeを解いたあとでcenterを解く

    # 柄付き
    state = puzzle["initial_state"].split(";")
    state, perm = run_moves(state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    state, c5moves = solve_center_fast(state, cube_size, legal_moves, perm,final_state)
    total_moves.extend(c5moves)

    # 最後に表面を始末
    state, smoves, perm = solve_final_surface(state, final_state, cube_size, 0, legal_moves, perm)
    total_moves.extend(smoves)
    
    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")

    print("solve333")
    
    alias_dict = {}
    for ist, ali in zip(initial_state, alias):
        alias_dict[ist] = ali
    
    state333 = []
    for s in state:
        state333.append(alias_dict[s])

    try:
        # moves333 = solve_by_solver(initial_state)
        if cube_size % 2 == 0:
            moves333 = solve_by_solver(state333, solver_path_444=solver_path_444)
        
        else:
            moves333 = solve_by_solver(state333, solver_path=solver_path)
    except Exception:
        print("solver error, skip solver")
        moves333 = []
    total_moves.extend(moves333)
    state, perm = run_moves(state, moves333, legal_moves, perm)
    
    print("change to original", len(total_moves))
    state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])

    suff = get_best_orient(state, final_state, legal_moves, cube_size)
    state, perm = run_moves(state, suff, legal_moves, perm)
    total_moves.extend(suff)
    
    print("score", len(total_moves))
    print(intmove, len(edgemoves), "edge", pmlen, tmlen, "(parity stat)",len(c5moves), "surf", len(smoves), "finalurf", len(moves333), "solve333")


    state, sumoves, perm = solve_surface_rotation(state, cube_size, perm, legal_moves, final_state)
    total_moves.extend(sumoves)

    print("validation", len(total_moves))
    test_final_state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    for i in range(6):
        print_surface(test_final_state, cube_size, i)
        print("------------------------")
    print("score", len(total_moves))
    with open(output_path ,"w") as f:
        f.write(".".join(total_moves))

    print("score", len(total_moves))
    print(intmove, len(edgemoves), "edge", pmlen, tmlen, "(parity stat)",len(c5moves), "surf", len(smoves), "finalurf", len(moves333), "solve333")
    
    score = 0
    for s, f in zip(state, final_state):
        if s == f:
            score += 1
    print(score, len(state))


    return state, total_moves, perm
    

def diff_wo_corner(state, final, cube_size) -> int:
    score = 0
    for i in range(6):
        for x in range(cube_size):
            for y in range(cube_size):
                if y == 0 or y == cube_size - 1:
                    continue
                    
                if x == 0 or x == cube_size - 1:
                    continue

                idx = cube_size ** 2 * i + y * cube_size + x
                if state[idx] != final[idx]:
                    score += 1
    return score


def diff_all(state, final, cube_size) -> int:
    score = 0
    for i in range(6):
        for x in range(cube_size):
            for y in range(cube_size):
                idx = cube_size ** 2 * i + y * cube_size + x
                if state[idx] != final[idx]:
                    score += 1
    return score

def solve_wildcard_wo_img(puzzles_path:str="data/puzzles.csv", sln_path:str="data/merged_122017.csv", puzzle_info_path:str="data/puzzle_info.csv", solver_path_444="data/merged_122017.csv", solver_path:str="/home/shiku/AI/kaggle/santa2023/rubiks-cube-NxNxN-solver", id:int=281, initial_state:str=None, initial_moves:str=None, output_path:str="sln.txt"):
    import pandas as pd
    tilewise = False
    puzzle = pd.read_csv(puzzles_path).set_index("id").loc[id]
    sln = pd.read_csv(sln_path).set_index("id").loc[id]
    sln_moves = sln["moves"].split(".")
    if initial_state is None:
        initial_state = puzzle["initial_state"].split(";")
    else:
        initial_state = initial_state.split(";")
    perm = [i for i in range(len(initial_state))]
    final_state = puzzle["solution_state"].split(";")
    cube_size = int(puzzle["puzzle_type"].split("/")[-1])
    wildcard = int(puzzle["num_wildcards"])
    legal_moves = get_moves(puzzle["puzzle_type"], puzzle_info_path)
    state = [s for s in initial_state]

    import matplotlib.pyplot as plt
    total_moves = []
    wildcard = 0
    if initial_moves is not None:
        imoves = initial_moves.split(".")
        ids = []
        diffs = []
            
        for p, mv in enumerate(imoves):
            state, perm = run_moves(state, [mv], legal_moves, perm)
            diff = diff_wo_corner(state, final_state, cube_size)
            print(p, diff, wildcard)
            ids.append(p)
            diffs.append(diff)
            if diff < wildcard:
                print("wildbreak", len(imoves), p)                
                imoves = imoves[:p+1]
                diffall = diff_all(state, final_state, cube_size)
                if diffall < wildcard:
                    print("oh solved...")
                    total_moves = imoves
                    with open(output_path ,"w") as f:
                        f.write(".".join(total_moves))
                        print("saved", output_path)

                    score = 0
                    for s, f in zip(state, final_state):
                        if s == f:
                            score += 1
                    print("score", len(total_moves))
                    print(score, len(state))
                    return
                    
                break
        plt.scatter(ids, diffs)
        plt.show()            
        total_moves.extend(imoves)

    else:
        intmove = 0

    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")


    state_333 = [s for s in state]
    ABCDEF = "ABCDEF"
    for i in range(6):
        for x in range(cube_size):
            for y in range(cube_size):
                if x == 0 or x == cube_size - 1:
                    if y == 0 or y == cube_size - 1:
                        continue
                idx = cube_size ** 2 * i + y * cube_size + x
                state_333[idx] = ABCDEF[i]
    

    for i in range(6):
        print_surface(state_333, cube_size, i)
        print("------------------------")



    try:
        if cube_size % 2 == 0:
            #pll = solve_oll(cube_size)
            #state, perm = run_moves(state, translate_move(pll, cube_size), legal_moves, perm)
            moves333 = solve_by_solver(state_333, solver_path_444=solver_path_444)
        else:
            moves333 = solve_by_solver(state_333, solver_path=solver_path)
    except Exception:
        moves333 = []
        print("solver error, skip solver")

    total_moves.extend(moves333)

    print("change to original", len(total_moves))
    state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])

    suff = get_best_orient(state, final_state, legal_moves, cube_size)
    state, perm = run_moves(state, suff, legal_moves, perm)
    total_moves.extend(suff)

    for i in range(6):
        print_surface(state, cube_size, i)
        print("------------------------")

    print("validation", len(total_moves))
    test_final_state, perm = run_moves(initial_state, total_moves, legal_moves, [i for i in range(len(initial_state))])
    for i in range(6):
        print_surface(test_final_state, cube_size, i)
        print("------------------------")
    
    score = 0
    for s, f in zip(test_final_state, final_state):
        if s == f:
            score += 1

    print("score", len(total_moves))
    print(score, len(state))

    if len(state) - score > wildcard:
        raise ValueError("validation failed")    

    with open(output_path ,"w") as f:
        f.write(".".join(total_moves))
        print("saved", output_path)
    


def solve_force_333(state:str, cube_size:int) -> List[str]:
    udict = {input_str[i] : ulist[i] for i in range(6)}

    def state2ubl(state, udict):
        state_split = state
        dim = int(np.sqrt(len(state_split) // 6))
        dim_2 = dim**2
        s = ''.join([udict[f] for f in state_split])
        slist = [s[:dim_2], s[2*dim_2:3*dim_2], s[dim_2:2*dim_2], s[5*dim_2:], s[4*dim_2:5*dim_2], s[3*dim_2:4*dim_2]]
        surf4 = ""
        for i in range(6):
            surf4 += slist[i][0:2]
            surf4 += slist[i][dim-1:dim+2]
            surf4 += slist[i][dim_2-dim-1:dim_2-dim+2]
            surf4 += slist[i][-1]
        return surf4

    state_for_solver = state2ubl(state, udict)
    print(state_for_solver)
    out = sv.solve(state_for_solver,15, 20)
    out = out.split(" ")
    output = []
    for o in out:
        print(o)
        ochar = ""
        if o[0] == "B":
            ochar = "-f*"
        elif o[0] == "F":
            ochar = "f0"
        elif o[0] == "L":
            ochar = "-r*"
        elif o[0] == "R":
            ochar = "r0"
        elif o[0] == "U":
            ochar = "-d*"
        elif o[0] == "D":
            ochar = "d0"
        else:
            continue
        if "'" in o:
            ochar = get_rev(ochar)
        if "2" in o:
            ochar += "x"
        if "3" in o:
            ochar = get_rev(ochar)
        output.append(ochar)
    return translate_move(output, cube_size)


def main():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--puzzles_path', type=str, default="data/puzzles.csv")
    args.add_argument('--sln_path', type=str, default="data/merged_122017.csv")
    args.add_argument('--puzzle_info_path', type=str, default="data/puzzle_info.csv")
    args.add_argument('--solver_path_444', type=str, default="TPR-4x4x4-Solver")
    args.add_argument('--solver_path', type=str, default="rubiks-cube-NxNxN-solver")
    args.add_argument('--id', type=int, default=281)
    args.add_argument('--initial_state')
    args.add_argument('--initial_moves')
    args.add_argument('--output_path', default="sln.txt")
    args.add_argument('--surf', action="store_true")
    args.add_argument('--js', action="store_true")
    args.add_argument('--add_minus', action="store_true")
    args.add_argument('--use_anl', action="store_true")
    args.add_argument('--skip_alias', action="store_true")
    args.add_argument('--wild', action="store_true")

    args = args.parse_args()
    if args.js:
        solve_only333(args.puzzles_path, args.sln_path, args.puzzle_info_path, args.solver_path_444, args.solver_path, args.id, args.initial_state, args.initial_moves, args.output_path)
    elif args.wild:
        solve_wildcard_wo_img(args.puzzles_path, args.sln_path, args.puzzle_info_path, args.solver_path_444, args.solver_path, args.id, args.initial_state, args.initial_moves, args.output_path)
    elif args.surf:
        solve_with_img(args.puzzles_path, args.sln_path, args.puzzle_info_path, args.solver_path_444, args.solver_path, args.id, args.initial_state, args.initial_moves, args.output_path)
    else:
        solve(args.puzzles_path, args.sln_path, args.puzzle_info_path, args.solver_path_444, args.solver_path, args.id, args.initial_state, args.initial_moves, args.output_path, use_anl=args.use_anl, skip_alias=args.skip_alias, force_skip_surf=False)
    
def test_333():

    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--puzzles_path', type=str, default="data/puzzles.csv")
    args.add_argument('--sln_path', type=str, default="data/merged_122017.csv")
    args.add_argument('--puzzle_info_path', type=str, default="data/puzzle_info.csv")
    args.add_argument('--solver_path_444', type=str, default="TPR-4x4x4-Solver")
    args.add_argument('--solver_path', type=str, default="rubiks-cube-NxNxN-solver")
    args.add_argument('--id', type=int, default=281)
    args.add_argument('--initial_state')
    args.add_argument('--initial_moves')
    args.add_argument('--output_path', default="sln.txt")
    args.add_argument('--surf', action="store_true")
    args.add_argument('--js', action="store_true")
    args.add_argument('--add_minus', action="store_true")
    args.add_argument('--use_anl', action="store_true")
    args.add_argument('--skip_alias', action="store_true")

    args = args.parse_args()
    
    puzzle = pd.read_csv(args.puzzles_path).set_index("id").loc[args.id]
    sln = pd.read_csv(args.sln_path).set_index("id").loc[args.id]
    sln_moves = sln["moves"].split(".")
    initial_state = puzzle["initial_state"].split(";")
    final_state = puzzle["solution_state"].split(";")
    cube_size = int(puzzle["puzzle_type"].split("/")[-1])
    legal_moves = get_moves(puzzle["puzzle_type"], args.puzzle_info_path)
    alias = get_alias(initial_state, sln_moves, legal_moves)
    total_moves = args.initial_moves.split(".")
    get_333_alias(args.id, initial_state, sln_moves, legal_moves, cube_size, total_moves, final_state, alias, output_path=args.output_path)
    

if __name__=="__main__":
    main()
    # test_333()