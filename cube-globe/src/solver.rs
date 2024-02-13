use std::{collections::{HashSet, VecDeque}, fs, path::Path};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::perm::{Permutation, SparsePermutation};
use crate::union_find::UnionFind;
use rayon::prelude::*;

pub struct Puzzle {
    initial_state: Vec<usize>,
    target_state: Vec<usize>,
    valid_moves: Vec<(String, Vec<usize>)>,
    puzzle_type_name: String,
    wildcard: i32,
}

impl Puzzle {
    pub fn new(initial_state: Vec<usize>, target_state: Vec<usize>, valid_moves: Vec<(String, Vec<usize>)>, puzzle_type_name: String, wildcard: i32) -> Puzzle {
        Puzzle { initial_state, target_state, valid_moves, puzzle_type_name, wildcard }
    }

    pub fn is_cube(&self) -> bool {
        self.puzzle_type_name.starts_with("cube")
    }
}

#[allow(unused)]
struct Scorer {
    score_map: Vec<Vec<i32>>,
}

#[allow(unused)]
const OPPOSITE: [usize; 6] = [5, 3, 4, 1, 2, 0];

#[allow(unused)]
impl Scorer {
    fn new(sz: usize, diff_penalty: i32, opposite_penalty: i32) -> Scorer {
        // TODO: this scorer ignores the target state
        let mut score_map = vec![];

        for i in 0..6 {
            for _ in 0..(sz * sz) {
                let mut row = vec![];
                for j in 0..6 {
                    if j == i {
                        row.push(0);
                    } else if j == OPPOSITE[i] {
                        row.push(diff_penalty + opposite_penalty);
                    } else {
                        row.push(diff_penalty);
                    }
                }
                score_map.push(row);
            }
        }

        Scorer { score_map }
    }

    fn compute_score(&self, state: &[usize]) -> i32 {
        assert_eq!(self.score_map.len(), state.len());

        let mut ret = 0;
        for i in 0..state.len() {
            ret += self.score_map[i][state[i]];
        }
        ret
    }

    fn score_diff_after_permutation(&self, state: &[usize], perm: &SparsePermutation) -> i32 {
        let mut ret = 0;
        for i in 0..perm.num_moves() {
            let src = perm.src(i);
            let dest = perm.dest(i);
            ret -= self.score_map[src][state[src]];
            ret += self.score_map[src][state[dest]];
        }
        ret
    }
}

#[derive(Clone)]
struct SimpleScorer {
    target: Vec<usize>,
}

impl SimpleScorer {
    fn new(target: Vec<usize>) -> SimpleScorer {
        SimpleScorer { target }
    }

    fn compute_score(&self, state: &[usize]) -> i32 {
        assert_eq!(self.target.len(), state.len());

        let mut ret = 0;
        for i in 0..state.len() {
            if state[i] != self.target[i] {
                ret += 1;
            }
        }
        ret
    }

    fn score_diff_after_permutation(&self, state: &[usize], perm: &SparsePermutation) -> i32 {
        let mut ret = 0;
        for i in 0..perm.num_moves() {
            let src = perm.src(i);
            let dest = perm.dest(i);
            if state[src] != self.target[src] {
                ret -= 1;
            }
            if state[dest] != self.target[src] {
                ret += 1;
            }
        }
        ret
    }

    // equivalent to score_diff_after_permutation(state, perm.conjugate(g)), but 4~5x times faster
    fn score_diff_after_permutation_conjugate(&self, state: &[usize], perm: &SparsePermutation, g: &Permutation) -> i32 {
        /*
        let mut ret = 0;
        for i in 0..perm.num_moves() {
            let src = g.perm_inv(perm.src(i));
            let dest = g.perm_inv(perm.dest(i));
            if state[src] != self.target[src] {
                ret -= 1;
            }
            if state[dest] != self.target[src] {
                ret += 1;
            }
        }
        */
        let mut ret = 0;
        unsafe {
            for i in 0..perm.num_moves() {
                let src = g.perm_inv_unchecked(perm.src_unchecked(i));
                let dest = g.perm_inv_unchecked(perm.dest_unchecked(i));
                if *state.get_unchecked(src) != *self.target.get_unchecked(src) {
                    ret -= 1;
                }
                if *state.get_unchecked(dest) != *self.target.get_unchecked(src) {
                    ret += 1;
                }
            }
        }
        ret
    }

    fn restrict_on(&self, size_after_restriction: usize, restrict_map: &[Option<usize>]) -> SimpleScorer {
        assert_eq!(self.target.len(), restrict_map.len());
        let mut target = vec![0; size_after_restriction];
        for i in 0..self.target.len() {
            if let Some(j) = restrict_map[i] {
                target[j] = self.target[i];
            }
        }
        SimpleScorer { target }
    }
}

fn enumerate_combo_search(seq: &mut Vec<usize>, res: &mut Vec<Vec<usize>>, cnt: &mut [i32], rem: usize) {
    if rem == 0 {
        for i in 0..cnt.len() {
            if cnt[i] != 0 {
                return;
            }
        }
        res.push(seq.clone());
        return;
    }

    let mut cur_penalty = 0;
    for i in 0..cnt.len() {
        cur_penalty += cnt[i].abs();
    }
    if cur_penalty as usize > rem {
        return;
    }

    for i in 0..cnt.len() {
        for d in [-1, 1] {
            let id = i * 2 + if d == -1 { 1 } else { 0 };
            if seq.len() > 0 && seq[seq.len() - 1] == (id ^ 1) {
                continue;
            }

            let new_penalty = cur_penalty - cnt[i].abs() + (cnt[i] + d).abs();
            if new_penalty as usize > rem - 1 {
                continue;
            }

            cnt[i] += d;
            seq.push(i * 2 + if d == -1 { 1 } else { 0 });
            enumerate_combo_search(seq, res, cnt, rem - 1);
            seq.pop();
            cnt[i] -= d;
        }
    }
}

fn enumerate_combo(num_cand: usize, len: usize) -> Vec<Vec<usize>> {
    let mut seq = vec![];
    let mut res = vec![];
    let mut cnt = vec![0; num_cand];
    enumerate_combo_search(&mut seq, &mut res, &mut cnt, len);
    res
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ComboCandidateData {
    asymmetry: Vec<Vec<usize>>,
    diagonal: Vec<Vec<usize>>,
    center_line: Vec<Vec<usize>>,
    edge_asymmetry: Vec<Vec<usize>>,
    edge_center: Vec<Vec<usize>>,
}

pub struct Solver {
    move_names: Vec<String>, 
    dense_moves: Vec<Permutation>,
    sparse_moves: Vec<SparsePermutation>,

    initial_state: Vec<usize>,
    target_state: Vec<usize>,
    extra_combo: Vec<Vec<usize>>,

    scorer: SimpleScorer,

    cube_size: usize,
    puzzle_type_name: String,
    center_only: bool,
    is_cube: bool,
    self_inverse: Vec<bool>,
    use_extended_combo: bool,
    wildcard: i32,
}

fn permutation_parity(mut seq1: Vec<usize>, seq2: Vec<usize>) -> bool {
    let mut parity = false;
    assert_eq!(seq1.len(), seq2.len());

    for i in 0..seq1.len() {
        if seq1[i] == seq2[i] {
            continue;
        }
        let mut found = false;
        for j in 1..seq1.len() {
            if seq2[i] == seq1[j] {
                seq1.swap(i, j);
                found = true;
                parity = !parity;
            }
        }
        assert!(found);
    }
    parity
}

fn forget_corner_color(state: &mut [usize], n: usize) {
    assert_eq!(state.len(), 6 * n * n);

    for i in 0..6 {
        state[i * n * n] = !0;
        state[i * n * n + (n - 1)] = !0;
        state[i * n * n + n * (n - 1)] = !0;
        state[(i + 1) * n * n - 1] = !0;
    }
}
fn reassign_edge_color(state: &mut [usize], n: usize) {
    assert_eq!(state.len(), 6 * n * n);

    let up = |i: usize| ((i * n * n + 1)..(i * n * n + n - 1)).collect::<Vec<_>>();
    let dw = |i: usize| ((i * n * n + n * (n - 1) + 1)..(i * n * n + n * n - 1)).collect::<Vec<_>>();
    let lf = |i: usize| {
        let mut ret = vec![];
        for j in 1..(n - 1) {
            ret.push(i * n * n + j * n);
        }
        ret
    };
    let rg = |i: usize| {
        let mut ret = vec![];
        for j in 1..(n - 1) {
            ret.push(i * n * n + j * n + n - 1);
        }
        ret
    };
    let mut pair = |a: Vec<usize>, b: Vec<usize>, rev: bool| {
        for i in 0..a.len() {
            let x = a[i];
            let y = b[if rev { a.len() - 1 - i } else { i }];
            let a = state[x];
            let b = state[y];
            state[x] = a + 6 * b;
            state[y] = b + 6 * a;
        }
    };

    pair(up(0), up(3), true);
    pair(lf(0), up(4), false);
    pair(rg(0), up(2), true);
    pair(dw(0), up(1), false);
    pair(lf(4), rg(3), false);
    pair(rg(4), lf(1), false);
    pair(dw(4), lf(5), true);
    pair(rg(1), lf(2), false);
    pair(dw(1), up(5), false);
    pair(rg(2), lf(3), false);
    pair(dw(2), rg(5), false);
    pair(dw(3), dw(5), true);

    // TODO: we may need a "garbage dump" but this is not compatible with parity checker
}

fn get_edge_center_stats(state: &[usize], n: usize) -> Vec<usize> {
    let up = |i: usize| i * n * n + n / 2;
    let dw = |i: usize| i * n * n + n * (n - 1) + n / 2;
    let lf = |i: usize| i * n * n + (n / 2) * n;
    let rg = |i: usize| i * n * n + (n / 2) * n + n - 1;

    let mut ret = vec![];
    let mut pair = |a, b, _| {
        let va: usize = state[a];
        let vb = state[b];
        ret.push(va.min(vb));
    };
    pair(up(0), up(3), true);
    pair(lf(0), up(4), false);
    pair(rg(0), up(2), true);
    pair(dw(0), up(1), false);
    pair(lf(4), rg(3), false);
    pair(rg(4), lf(1), false);
    pair(dw(4), lf(5), true);
    pair(rg(1), lf(2), false);
    pair(dw(1), up(5), false);
    pair(rg(2), lf(3), false);
    pair(dw(2), rg(5), false);
    pair(dw(3), dw(5), true);

    ret
}

fn get_face_center_stats(state: &[usize], n: usize) -> Vec<usize> {
    (0..6).map(|i| state[i * n * n + (n * n) / 2]).collect()
}

impl Solver {
    pub fn new(puzzle: Puzzle, center_only: bool, use_surface_move: bool, no_forget_corner: bool) -> Solver {
        let is_cube = puzzle.is_cube();
        let n = puzzle.initial_state.len();
        let mut sz: Option<usize> = None;
        for i in 1..=33 {
            if n == i * i * 6 {
                sz = Some(i);
            }
        }
        let sz = if is_cube { sz.unwrap() } else { /* dummy */ !0 };

        let mut move_names = vec![];
        let mut dense_moves = vec![];
        let mut sparse_moves = vec![];

        let mut keep_indices: Vec<Option<usize>> = vec![];
        if center_only {
            let mut idx = 0;
            for _ in 0..6 {
                for y in 0..sz {
                    for x in 0..sz {
                        if y == 0 || y == sz - 1 || x == 0 || x == sz - 1 {
                            keep_indices.push(None);
                        } else {
                            keep_indices.push(Some(idx));
                            idx += 1;
                        }
                    }
                }
            }
        } else {
            keep_indices = (0..n).map(Some).collect();
        }

        for (name, mv) in puzzle.valid_moves {
            if center_only && !use_surface_move {
                let id = name[1..].parse::<usize>().unwrap();
                if id == 0 || id == sz - 1 {
                    continue;
                }
            }

            let mut mv2 = vec![];
            for i in 0..mv.len() {
                assert_eq!(keep_indices[i].is_some(), keep_indices[mv[i]].is_some());
                if keep_indices[i].is_some() {
                    mv2.push(keep_indices[mv[i]].unwrap());
                }
            }

            let dense_move = Permutation::new(mv2);
            let sparse_move = dense_move.sparsify();
            move_names.push(name.clone());
            dense_moves.push(dense_move.clone());
            sparse_moves.push(sparse_move.clone());

            move_names.push("-".to_owned() + &name);
            dense_moves.push(!dense_move);
            sparse_moves.push(!sparse_move);
        }

        let mut initial_state = vec![];
        for i in 0..puzzle.initial_state.len() {
            if keep_indices[i].is_some() {
                initial_state.push(puzzle.initial_state[i]);
            }
        }
        let mut target_state = vec![];
        for i in 0..puzzle.target_state.len() {
            if keep_indices[i].is_some() {
                target_state.push(puzzle.target_state[i]);
            }
        }
        let mut wildcard = puzzle.wildcard;
        if !center_only {
            if puzzle.initial_state.iter().max().unwrap() == &5 {
                reassign_edge_color(&mut initial_state, sz);
                reassign_edge_color(&mut target_state, sz);
                if !no_forget_corner {
                    forget_corner_color(&mut target_state, sz);
                    wildcard += 24;
                }
            }
        }

        let self_inverse = dense_moves.iter().map(|x| (x * x).sparsify().num_moves() == 0).collect();

        let sz = sz - if center_only { 2 } else { 0 };

        Solver {
            move_names, dense_moves, sparse_moves,
            initial_state,
            target_state: target_state.clone(),
            scorer: SimpleScorer::new(target_state),
            cube_size: sz,
            puzzle_type_name: puzzle.puzzle_type_name,
            center_only,
            extra_combo: vec![],
            is_cube,
            self_inverse,
            use_extended_combo: false,
            wildcard,
        }
    }

    pub fn set_use_extended_combo(&mut self)  {
        self.use_extended_combo = true;
    }

    pub fn solution_to_string(&self, sol: &[usize]) -> String {
        let mut ret = vec![];
        for i in 0..sol.len() {
            if i != 0 {
                ret.push(String::from("."));
            }
            ret.push(self.move_names[sol[i]].clone());
        }
        ret.concat()
    }

    fn simplify(&self, seq: &[usize]) -> Vec<usize> {
        let mut ret = vec![];
        for &p in seq {
            if ret.len() > 0 && (ret[ret.len() - 1] == p ^ 1 || (ret[ret.len() - 1] == p && self.self_inverse[p])) {
                ret.pop();
            } else {
                ret.push(p);
            }
        }
        ret
    }
    
    fn simplify_profit(&self, left: &[usize], right: &[usize]) -> usize {
        let m = left.len().min(right.len());
        for i in 0..m {
            if !(left[left.len() - 1 - i] == (right[i] ^ 1) || (self.self_inverse[right[i]] && left[left.len() - 1 - i] == right[i])) {
                return i;
            }
        }
        m
    }
    
    fn simplify_profit3(&self, left: &[usize], mid: &[usize], right: &[usize]) -> usize {
        (self.simplify_profit(left, mid) + self.simplify_profit(mid, right)).min(mid.len())
    }
        
    // greedy algorithm allowing adding moves only to the end of the sequence
    pub fn greedy(&self) -> Vec<usize> {
        let mut ret = vec![];
        let mut state = self.initial_state.clone();
        loop {
            let score = self.scorer.compute_score(&state);
            let mut best_move: Option<(usize, i32)> = None;

            for i in 0..self.sparse_moves.len() {
                let mv = &self.sparse_moves[i];
                let score2 = score + self.scorer.score_diff_after_permutation(&state, mv);

                if score <= score2 {
                    continue;
                }

                match best_move {
                    None => best_move = Some((i, score2)),
                    Some((_, s)) => if s > score2 {
                        best_move = Some((i, score2));
                    }
                }
            }

            match best_move {
                None => break,
                Some((id, s)) => {
                    eprintln!("{} -> {}", score, s);
                    ret.push(id);
                    self.sparse_moves[id].apply_inplace(&mut state);
                }
            }
        }

        ret
    }

    fn run_greedy2<R: Rng>(&self, mut seq: Vec<usize>, combo_sequences: &[Vec<usize>], reduction_threshold: i32, wildcard_mode: bool, rng: &mut R) -> Vec<usize> {
        let mut state = self.initial_state.clone();

        for &mv in &seq {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }

        let mut combo_perms = vec![];
        for seq in combo_sequences {
            assert!(seq.len() > 0);
            let mut perm = self.dense_moves[seq[0]].clone();
            for i in 1..seq.len() {
                perm = &self.dense_moves[seq[i]] * perm;
            }
            combo_perms.push(perm.sparsify());
        }
        let mut num_perturbation = 0;

        loop {
            let score = self.scorer.compute_score(&state);

            let mut cumulative_moves = vec![Permutation::identity(self.initial_state.len())];
            for i in 0..seq.len() {
                cumulative_moves.push(&cumulative_moves[i] * &self.dense_moves[seq[seq.len() - 1 - i]]);
            }
            cumulative_moves.reverse();

            let num_shards = 32;
            let best_moves_per_shard = (0..num_shards).into_par_iter().map(|shard_id| {
                let shard_start = shard_id * combo_perms.len() / num_shards;
                let shard_end = (shard_id + 1) * combo_perms.len() / num_shards;

                let mut best_move_profit = -100;
                let mut best_moves = vec![];

                for p in 0..=seq.len() {
                    for i in shard_start..shard_end {
                        let score2 = score + self.scorer.score_diff_after_permutation_conjugate(&state, &combo_perms[i], &cumulative_moves[p]);
    
                        let profit;
                        if wildcard_mode {
                            if score2 > self.wildcard {
                                continue;
                            }
                            let seq_penalty = combo_sequences[i].len() as i32 - self.simplify_profit3(&seq[..p], &combo_sequences[i], &seq[p..]) as i32 * 2;
                            if seq_penalty >= 0 {
                                continue;
                            }
                            profit = (score - score2) * 6 - seq_penalty as i32;
                        } else {
                            if self.is_cube {
                                profit = score - score2;

                                if score - reduction_threshold <= score2 {  // TODO: make this param configurable
                                    continue;
                                }
                            } else {
                                let seq_penalty = combo_sequences[i].len() as i32 - self.simplify_profit3(&seq[..p], &combo_sequences[i], &seq[p..]) as i32 * 2;
                                profit = (score - score2) * 2 - seq_penalty as i32;

                                if !(score > score2 || (score == score2 && seq_penalty < 0)) {
                                    continue;
                                }
                            }
                        }

                        if best_move_profit < profit {
                            best_move_profit = profit;
                            best_moves.clear();
                            best_moves.push((p, i, score2));
                        } else if best_move_profit == profit {
                            best_moves.push((p, i, score2));
                        }
                    }
                }

                (best_move_profit, best_moves)
            }).collect::<Vec<_>>();

            let mut best_move_profit = -100;
            let mut best_moves = vec![];

            for (p, m) in best_moves_per_shard {
                if m.is_empty() {
                    continue;
                }
                if best_move_profit < p {
                    best_move_profit = p;
                    best_moves = m;
                } else if best_move_profit == p {
                    best_moves.extend(m);
                }
            }

            if best_moves.is_empty() {
                break;
            }
            let (pos, id, s) = best_moves[rng.gen_range(0..best_moves.len())];
            let mut seq2: Vec<usize> = vec![];
            seq2.extend(&seq[..pos]);
            seq2.extend(&combo_sequences[id]);
            seq2.extend(&seq[pos..]);
            // sol = sol2;
            let len_before = seq.len();
            seq = self.simplify(&seq2);
            eprintln!("{} -> {} ({:?}, act. cost = {})", score, s, combo_sequences[id], seq.len() as i32 - len_before as i32);

            combo_perms[id].conjugate(&cumulative_moves[pos]).apply_inplace(&mut state);
        }

        seq
    }

    fn globe_center_fix(&self) -> Vec<usize> {
        if self.is_cube {
            return vec![];
        }

        let byname = |x: &str| {
            for i in 0..self.move_names.len() {
                if &self.move_names[i] == x {
                    return i;
                }
            }
            panic!();
        };

        let mut rmax = 0;
        let mut fmax = 0;
        for name in &self.move_names {
            if &name[0..1] == "r" {
                rmax = rmax.max(name[1..].parse().unwrap());
            }
            if &name[0..1] == "f" {
                fmax = fmax.max(name[1..].parse().unwrap());
            }
        }
        
        if rmax % 2 == 1 {
            return vec![];
        }

        let mid = byname(&format!("r{}", rmax / 2));
        let mid_move = &self.sparse_moves[mid];
        let mut mid_targets = vec![];
        for i in 0..mid_move.num_moves() {
            mid_targets.push(mid_move.src(i));
        }

        {
            let mut state = self.initial_state.clone();
            for i in 0..=(fmax / 2 + 1) {
                let mut isok = true;
                for j in 0..mid_targets.len() {
                    if state[mid_targets[j]] != self.target_state[mid_targets[j]] {
                        isok = false;
                        break;
                    }
                }
                if isok {
                    return vec![mid; i];
                }
                mid_move.apply_inplace(&mut state);
            }
        }
        {
            let mut state = self.initial_state.clone();
            for i in 0..=(fmax / 2 + 1) {
                let mut isok = true;
                for j in 0..mid_targets.len() {
                    if state[mid_targets[j]] != self.target_state[mid_targets[j]] {
                        isok = false;
                        break;
                    }
                }
                if isok {
                    return vec![mid ^ 1; i];
                }
                mid_move.apply_inv_inplace(&mut state);
            }
        }
        panic!();
    }

    pub fn globe_resolve_parity(&self, mut seq: Vec<usize>) -> Vec<usize> {
        let byname = |x: &str| {
            for i in 0..self.move_names.len() {
                if &self.move_names[i] == x {
                    return i;
                }
            }
            panic!();
        };

        let mut hoge = self.target_state.clone();
        hoge.sort();
        for i in 1..hoge.len() {
            if hoge[i] == hoge[i - 1] {
                // parity does not matter
                return seq;
            }
        }

        eprintln!("run parity resolution");
        let mut rmax = 0;
        let mut fmax = 0;
        for name in &self.move_names {
            if &name[0..1] == "f" {
                fmax = fmax.max(name[1..].parse().unwrap());
            }
            if &name[0..1] == "r" {
                rmax = rmax.max(name[1..].parse().unwrap());
            }
        }

        let num_r = rmax + 1;
        let num_f = fmax + 1;
        assert_eq!(num_f % 2, 0);
        assert_eq!(num_r * num_f, self.initial_state.len());

        let num_r_groups = num_r / 2;

        let compute_parity = |state: &[usize]| {
            let mut parity = vec![];
            for i in 0..num_r_groups {
                let mut seq1 = vec![];
                let mut seq2 = vec![];
                for r in [i, rmax - i] {
                    for j in 0..num_f {
                        let id = r * num_f + j;
                        seq1.push(state[id]);
                        seq2.push(self.target_state[id]);
                    }
                }
                parity.push(permutation_parity(seq1, seq2));
            }
            parity    
        };

        let mut state = self.initial_state.clone();
        for &mv in &seq {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }

        let mut parity = compute_parity(&state);
        eprintln!("{:?}", parity);

        if num_f % 4 == 2 {
            // hemisphere rotation switches all parities
            let mut n_true = 0;
            let mut n_false = 0;
            for i in 0..parity.len() {
                if parity[i] {
                    n_true += 1;
                } else {
                    n_false += 1;
                }
            }
            if n_true > n_false {
                let mv = byname(&format!("f0"));
                seq.push(mv);
                self.sparse_moves[mv].apply_inplace(&mut state);
                for i in 0..parity.len() {
                    parity[i] = !parity[i];
                }
            }
        }

        for i in 0..num_r_groups {
            if parity[i] {
                let mv = byname(&format!("r{}", i));
                seq.push(mv);
                self.sparse_moves[mv].apply_inplace(&mut state);
            }
        }

        let new_parity = compute_parity(&state);
        for i in 0..new_parity.len() {
            assert!(!new_parity[i]);
        }

        seq
    }

    // greedy algorithm allowing inserting moves in the middle of the sequence
    pub fn greedy2(&self, seed: u64, perturb: Option<usize>, reduction_threshold_cube: i32) -> Vec<usize> {
        let mut rng = SmallRng::seed_from_u64(seed);

        let orig_sequences = (0..self.dense_moves.len()).map(|x| vec![x]).collect::<Vec<_>>();

        let initial_seq;
        if let Some(perturb) = perturb {
            let mut seq = self.globe_center_fix();
            for _ in 0..perturb {
                seq.push(rng.gen_range(0..self.dense_moves.len()));
            }
            initial_seq = seq;
        } else {
            initial_seq = self.globe_center_fix();
        }
        let ret = self.run_greedy2(initial_seq, &orig_sequences, 0, false, &mut rng);

        let ret = if self.is_cube { ret } else { self.globe_resolve_parity(ret) };

        let mut combo_sequences = vec![];
        for i in 0..self.dense_moves.len() {
            combo_sequences.push(vec![i]);

            for j in 0..self.dense_moves.len() {
                if i == j {
                    continue;
                }
                combo_sequences.push(vec![i, j, i ^ 1, j ^ 1]);
            }
        }
        combo_sequences.extend(self.globe_combo());
        for extra_combo_ in &self.extra_combo {
            for inv in [false, true] {
                let extra_combo;

                if inv {
                    let mut e = vec![];
                    for i in 0..extra_combo_.len() {
                        e.push(extra_combo_[extra_combo_.len() - 1 - i] ^ 1);
                    }
                    extra_combo = e;
                } else {
                    extra_combo = extra_combo_.clone();
                }

                let mut perm = Permutation::identity(self.initial_state.len());
                for i in 0..extra_combo.len() {
                    perm = &self.dense_moves[extra_combo[i]] * perm;
                }
                let sparse_perm = perm.sparsify();

                if !self.is_cube && sparse_perm.num_moves() != 4 {
                    // ad hoc
                    continue;
                }
                for j in 0..self.dense_moves.len() {
                    let mut combo_sequence = vec![j];
                    combo_sequence.extend(&extra_combo);
                    combo_sequence.push(j ^ 1);

                    combo_sequences.push(self.simplify(&combo_sequence));
                }
                combo_sequences.push(extra_combo.clone());
            }
        }
        if !self.is_cube {
            // remove odd-length combo
            let mut combo_sequences2 = vec![];
            for c in combo_sequences {
                if c.len() % 2 == 0 {
                    combo_sequences2.push(c);
                }
            }
            combo_sequences = combo_sequences2;
        }

        let reduction_threshold = if self.is_cube { reduction_threshold_cube } else { 0 };
        let mut ret = if self.is_cube {
            self.run_greedy2(ret, &combo_sequences, reduction_threshold, false, &mut rng)
        } else {
            self.anneal(ret, &combo_sequences, None, 1, &mut rng).0
        };

        if self.puzzle_type_name.starts_with("globe") {
            let score = self.evaluate(&ret);
            if score > 0 {
                let mut hoge = self.target_state.clone();
                hoge.sort();
                let mut alldiff = true;
                for i in 1..hoge.len() {
                    if hoge[i] == hoge[i - 1] {
                        // parity does not matter
                        alldiff = false;
                    }
                }
                if alldiff {
                    // hard instances may require 3-piece replacement
                    ret = self.three_piece(ret);
                }
            }
        }
        ret
    }

    fn combo_cache_path(&self, name: &str) -> String {
        format!("cache/{}", name)
    }

    fn move_ids_for_bucket(&self, a: usize) -> Vec<usize> {
        let num_buckets = (self.cube_size + 1) / 2;
        assert_eq!(self.sparse_moves.len() % 6, 0);
        let move_per_axis = self.sparse_moves.len() / 6;

        let mut ret = vec![];
        for i in 0..3 {
            for j in 0..move_per_axis {
                let id = i * move_per_axis * 2 + j * 2;
                if self.center_only {
                    if j == 0 || j == move_per_axis - 1 {
                        continue;
                    }

                    let bucket = (j - 1).min(move_per_axis - 2 - j);
                    if bucket == a {
                        ret.push(id);
                        ret.push(id ^ 1);
                    }
                } else {
                    let bucket = j.min(move_per_axis - 1 - j);
                    if bucket == a {
                        ret.push(id);
                        ret.push(id ^ 1);
                    }
                }
            }
        }
        ret
    }

    fn move_ids_for_combo_bucket(&self, a: usize, b: usize) -> Vec<usize> {
        let num_buckets = (self.cube_size + 1) / 2;
        assert_eq!(self.sparse_moves.len() % 6, 0);
        let move_per_axis = self.sparse_moves.len() / 6;

        let mut move_buckets = vec![vec![]; num_buckets];
        let mut surfaces = vec![];

        for i in 0..3 {
            for j in 0..move_per_axis {
                let id = i * move_per_axis * 2 + j * 2;
                if self.center_only {
                    if j == 0 || j == move_per_axis - 1 {
                        surfaces.push(id);
                        surfaces.push(id ^ 1);
                        continue;
                    }

                    let bucket = (j - 1).min(move_per_axis - 2 - j);
                    move_buckets[bucket].push(id);
                    move_buckets[bucket].push(id ^ 1);
                } else {
                    let bucket = j.min(move_per_axis - 1 - j);
                    move_buckets[bucket].push(id);
                    move_buckets[bucket].push(id ^ 1);
                }
            }
        }

        let mut move_ids: Vec<usize> = vec![];
        if self.center_only {
            move_ids.extend(&surfaces);
        } else {
            move_ids.extend(&move_buckets[0]);
        }
        if self.center_only || a != 0 {
            move_ids.extend(&move_buckets[a]);
        }
        if a != b && (self.center_only || b != 0) {
            move_ids.extend(&move_buckets[b]);
        }

        move_ids
    }

    fn bucket_cells(&self, a: usize, b: usize) -> Vec<usize> {
        let mut ret = vec![];
        for i in 0..6 {
            for y in 0..self.cube_size {
                for x in 0..self.cube_size {
                    let id = i * self.cube_size * self.cube_size + y * self.cube_size + x;
                    let y_id = y.min(self.cube_size - 1 - y);
                    let x_id = x.min(self.cube_size - 1 - x);

                    if y_id <= x_id {
                        if (y_id, x_id) == (a, b) {
                            ret.push(id);
                        }
                    } else {
                        if (x_id, y_id) == (a, b) {
                            ret.push(id);
                        }
                    }
                }
            }
        }
        ret
    }

    fn enumerate_combo(&self) -> ComboCandidateData {
        if !self.puzzle_type_name.starts_with("cube") {
            panic!("this solver currently supports cube instances only");
        }

        let combo_cache_path = self.combo_cache_path("cube");

        if Path::new(&combo_cache_path).exists() {
            eprintln!("cache exists at {}", combo_cache_path);
            let json = std::fs::read_to_string(&combo_cache_path).unwrap();
            let data: ComboCandidateData = serde_json::from_str(&json).unwrap();

            return data;
        }
        if self.cube_size % 2 == 0 {
            panic!("to populate the cache, first run the solver with odd-sized cube");
        }
        if self.center_only {
            panic!("to populate the cache, first run the solver without --center-only");
        }

        eprintln!("cache does not exist at {}; compute combos", combo_cache_path);

        let mut cell_bucket = vec![];
        for _ in 0..6 {
            for y in 0..self.cube_size {
                for x in 0..self.cube_size {
                    let y_id = y.min(self.cube_size - 1 - y);
                    let x_id = x.min(self.cube_size - 1 - x);

                    if y_id <= x_id {
                        cell_bucket.push((y_id, x_id));
                    } else {
                        cell_bucket.push((x_id, y_id));
                    }
                }
            }
        }

        assert_eq!(self.sparse_moves.len() % 6, 0);

        let mut cand_data = ComboCandidateData {
            asymmetry: vec![],
            diagonal: vec![],
            center_line: vec![],
            edge_asymmetry: vec![],
            edge_center: vec![],
        };

        for mode in 0..5 {
            let a;
            let b;

            if mode == 0 {
                // asymmetry
                eprintln!("Mode asymmetry");
                a = 1;
                b = 2;
            } else if mode == 1 {
                // diagonal
                eprintln!("Mode diagonal");
                a = 1;
                b = 1;
            } else if mode == 2 {
                eprintln!("Mode center");
                a = 1;
                b = self.cube_size / 2;
                if self.cube_size % 2 == 0 {
                    continue;
                }
            } else if mode == 3 {
                eprintln!("Mode edge_asymmetry");
                a = 0;
                b = 1;
            } else if mode == 4 {
                eprintln!("Mode edge_center");
                a = 0;
                b = self.cube_size / 2;
            } else {
                unreachable!();
            }

            // reduce the size of the symmetry group for combo verification
            let mut restrict_map = vec![];
            let mut size_after_restriction = 0;
            for i in 0..self.initial_state.len() {
                let cb = cell_bucket[i];
                if cb.0 == a || cb.0 == b || cb.1 == a || cb.1 == b {
                    restrict_map.push(Some(size_after_restriction));
                    size_after_restriction += 1;
                } else {
                    restrict_map.push(None);
                }
            }
            eprintln!("size of the symmetry group for combo verification: {}", size_after_restriction);

            let mut target_restriction = vec![];
            let mut target_restriction_size = 0;
            for i in 0..self.initial_state.len() {
                if cell_bucket[i] == (a, b) {
                    assert!(restrict_map[i].is_some());
                    target_restriction.push(Some(target_restriction_size));
                    target_restriction_size += 1;
                } else {
                    if restrict_map[i].is_some() {
                        target_restriction.push(None);
                    }
                }
            }

            let move_ids = self.move_ids_for_combo_bucket(a, b);

            assert_eq!(move_ids.len() % 2, 0);

            let mut combos = enumerate_combo(move_ids.len() / 2, 8);
            if mode < 3 {
                let combos2 = enumerate_combo(move_ids.len() / 2, 4);
                combos.extend(combos2);
            }
            eprintln!("# combo candidates: {}", combos.len());

            let mut restricted_moves = vec![];
            for &id in &move_ids {
                restricted_moves.push(self.dense_moves[id].restrict_on(size_after_restriction, &restrict_map));
            }

            eprintln!("Start testing combos");

            let num_proc = 32;
            let mut handles = vec![];
            let valid_combos = std::sync::Arc::new(std::sync::Mutex::<Vec<Vec<usize>>>::new(vec![]));

            let start = std::time::Instant::now();
            for i in 0..num_proc {
                let valid_combos = valid_combos.clone();
                let shard_start = i * combos.len() / num_proc;
                let shard_end = (i + 1) * combos.len() / num_proc;
                let combos = combos[shard_start..shard_end].to_owned();
                let restricted_moves = restricted_moves.clone();
                let target_restriction = target_restriction.clone();

                let handle = std::thread::spawn(move || {
                    let mut valid_combos_local = vec![];
                    for combo in &combos {
                        let mut mv = restricted_moves[combo[0]].clone();
                        for j in 1..combo.len() {
                            mv = &restricted_moves[combo[j]] * mv;
                        }
                        if mv.is_restriction_faithful(target_restriction_size, &target_restriction) {
                            let n = mv.sparsify().num_moves();
                            let n_limit = if mode >= 3 || combo.len() == 4 { 6 } else { 5 };
                            if n == 0 || n > n_limit {
                                continue;
                            }
                            valid_combos_local.push(combo.clone());
                        }
                    }
                    let mut valid_combos = valid_combos.lock().unwrap();
                    valid_combos.extend(valid_combos_local);
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
            let valid_combos = valid_combos.lock().unwrap().clone();
            eprintln!("done in {}[s], {} valid combos", start.elapsed().as_secs_f64(), valid_combos.len());

            // now test with the entire sym group
            let mut target_restriction = vec![];
            let mut target_restriction_size = 0;
            for i in 0..self.initial_state.len() {
                if cell_bucket[i] == (a, b) {
                    target_restriction.push(Some(target_restriction_size));
                    target_restriction_size += 1;
                } else {
                    target_restriction.push(None);
                }
            }

            let mut combo_permutations = HashSet::new();
            let mut unique_combos = vec![];
            for valid_combo in valid_combos {
                let mut mv = self.dense_moves[move_ids[valid_combo[0]]].clone();
                for i in 1..valid_combo.len() {
                    mv = &self.dense_moves[move_ids[valid_combo[i]]] * mv;
                }
                assert!(mv.is_restriction_faithful(target_restriction_size, &target_restriction));

                if combo_permutations.insert(mv.clone()) {
                    unique_combos.push(valid_combo);
                }
            }

            eprintln!("{} unique combos", unique_combos.len());

            if mode == 0 {
                cand_data.asymmetry = unique_combos;
            } else if mode == 1 {
                cand_data.diagonal = unique_combos;
            } else if mode == 2 {
                cand_data.center_line = unique_combos;
            } else if mode == 3 {
                cand_data.edge_asymmetry = unique_combos;
            } else if mode == 4 {
                cand_data.edge_center = unique_combos;
            }
        }

        eprintln!("write candidate data to cache at {}", combo_cache_path);
        let json = serde_json::to_string(&cand_data).unwrap();
        fs::write(&combo_cache_path, &json).unwrap();

        cand_data
    }

    fn solve_local_many(&self, sol: Vec<usize>, cand_data: &Vec<Vec<usize>>, a: usize, b: usize, seed: u64, n_trials: i32) -> (Vec<usize>, i32) {
        eprintln!("* solve_local_many {} {}", a, b);
        eprintln!("consider {} combo", cand_data.len());

        // eprintln!("* solve_local {} {}", a, b);
        assert!(a <= b);
        let move_ids = self.move_ids_for_combo_bucket(a, b);
        let related_cells = self.bucket_cells(a, b);

        let size_after_restriction = related_cells.len();
        let mut restrict_map = vec![None; self.initial_state.len()];
        for i in 0..related_cells.len() {
            restrict_map[related_cells[i]] = Some(i);
        }

        let mut sparse_moves = vec![];
        let mut dense_moves = vec![];
        for i in 0..self.sparse_moves.len() {
            let sparse_move = self.sparse_moves[i].restrict_on(size_after_restriction, &restrict_map);
            let dense_move = sparse_move.densify();

            sparse_moves.push(sparse_move);
            dense_moves.push(dense_move);
        }

        let mut combo_sequences = vec![];
        let mut combo_perms = vec![];
        for combo in cand_data {
            let mut ids = vec![];
            let mut mv = Permutation::identity(self.initial_state.len());
            for &i in combo {
                ids.push(move_ids[i]);
                mv = &self.dense_moves[move_ids[i]] * mv;
            }

            combo_sequences.push(ids);
            assert!(mv.is_restriction_faithful(size_after_restriction, &restrict_map));
            combo_perms.push(mv.restrict_on(size_after_restriction, &restrict_map).sparsify());
        }

        let scorer = self.scorer.restrict_on(size_after_restriction, &restrict_map);
        let mut state = vec![0; size_after_restriction];
        for i in 0..related_cells.len() {
            state[i] = self.initial_state[related_cells[i]];
        }

        for &mv in &sol {
            sparse_moves[mv].apply_inplace(&mut state);
        }

        let ids = (0..n_trials).collect::<Vec<_>>().par_iter().map(|&i| {
            let mut state = state.clone();
            self.solve_local(sol.clone(), &scorer, &mut state, size_after_restriction, &dense_moves, &combo_sequences, &combo_perms, seed + i as u64)
        }).collect::<Vec<_>>();

        let mut best_seq = vec![];
        let mut best = (10000000, 0);
        for (seq, score) in ids {
            let p = (score, seq.len());
            if p < best {
                best = p;
                best_seq = seq;
            }
        }

        (best_seq, best.0)
    }

    #[allow(unused)]
    fn solve_local2(&self, sol: Vec<usize>, cand_data: &Vec<Vec<usize>>, a: usize, b: usize, seed: u64) -> (Vec<usize>, i32) {
        assert!(a <= b);

        let related_cells = self.bucket_cells(a, b);

        let mut restrict_map = vec![None; self.initial_state.len()];
        for i in 0..related_cells.len() {
            restrict_map[related_cells[i]] = Some(i);
        }

        let move_ids = self.move_ids_for_combo_bucket(a, b);
        let mut combo_sequences = vec![];
        for combo in cand_data {
            let mut ids = vec![];
            for &i in combo {
                ids.push(move_ids[i]);
            }

            combo_sequences.push(ids);
        }

        let mut rng = SmallRng::seed_from_u64(seed);
        self.anneal(sol, &combo_sequences, Some(&restrict_map), 1, &mut rng)
    }

    fn anneal<R: Rng>(&self, sol: Vec<usize>, allowed_sequences: &[Vec<usize>], restrict_map: Option<&[Option<usize>]>, eff: i32, rng: &mut R) -> (Vec<usize>, i32) {
        use rand::seq::SliceRandom;

        let mut n = 0;  // number of tiles to be considered
        if let Some(restrict_map) = restrict_map {
            for i in 0..restrict_map.len() {
                if restrict_map[i].is_some() {
                    n += 1;
                }
            }
        } else {
            n = self.initial_state.len();
        }

        let sparse_moves;
        let dense_moves;
        let scorer;
        let mut state;

        if let Some(restrict_map) = restrict_map {
            let mut s = vec![];
            let mut d = vec![];
            for i in 0..self.sparse_moves.len() {
                let sparse_move = self.sparse_moves[i].restrict_on(n, &restrict_map);
                let dense_move = sparse_move.densify();

                s.push(sparse_move);
                d.push(dense_move);
            }
            sparse_moves = s;
            dense_moves = d;

            scorer = self.scorer.restrict_on(n, &restrict_map);
            state = vec![0; n];
            for i in 0..self.initial_state.len() {
                if let Some(j) = restrict_map[i] {
                    state[j] = self.initial_state[i];
                }
            }
        } else {
            sparse_moves = self.sparse_moves.clone();
            dense_moves = self.dense_moves.clone();

            scorer = self.scorer.clone();
            state = self.initial_state.clone();
        }

        for &mv in &sol {
            sparse_moves[mv].apply_inplace(&mut state);
        }

        let mut allowed_perms = vec![];
        for seq in allowed_sequences {
            let mut mv = Permutation::identity(self.initial_state.len());
            for &i in seq {
                mv = &self.dense_moves[i] * mv;
            }

            if let Some(restrict_map) = restrict_map {
                assert!(mv.is_restriction_faithful(n, &restrict_map));
                allowed_perms.push(mv.restrict_on(n, &restrict_map).sparsify());
            } else {
                allowed_perms.push(mv.sparsify());
            }
        }

        let num_shards = 256;
        let mut seeds = vec![];
        for _ in 0..num_shards {
            seeds.push(rng.next_u64());
        }

        let beam_width = 64;

        // (total profit, score, state, sequence)
        let initial_score = scorer.compute_score(&state);
        let mut least_score = (sol.clone(), initial_score);
        let mut score_zero_best: Option<Vec<usize>> = None;

        let mut candidates = vec![(0, initial_score, state, sol)];

        while !candidates.is_empty() {
            eprintln!("===== beam =====");
            eprintln!("best beam score: {}, len: {}", candidates[0].1, candidates[0].3.len());
            // for cand in &candidates {
            //     eprintln!("score {}, seq len + {} {:?}", cand.1, cand.3.len() as i32 - init_seq_len as i32, cand.2);
            // }
            for cand in &candidates {
                if least_score.1 > cand.1 {
                    least_score = (cand.3.clone(), cand.1);
                }
            }

            let mut cumulative_moves_all = vec![];

            for i in 0..candidates.len() {
                let sol = &candidates[i].3;
                let mut cumulative_moves = vec![Permutation::identity(n)];
                for j in 0..sol.len() {
                    cumulative_moves.push(&cumulative_moves[j] * &dense_moves[sol[sol.len() - 1 - j]]);
                }
                cumulative_moves.reverse();
                cumulative_moves_all.push(cumulative_moves);
            }

            let cands = (0..num_shards).into_par_iter().map(|shard_id| {
                let shard_start = shard_id * allowed_sequences.len() / num_shards;
                let shard_end = (shard_id + 1) * allowed_sequences.len() / num_shards;

                let mut best_profit = -100;
                let mut best_profit_cands = vec![];

                for c in 0..candidates.len() {
                    let (total_profit, _, state, sol) = &candidates[c];
                    let total_profit = *total_profit;

                    for p in 0..=sol.len() {
                        for i in shard_start..shard_end {
                            let score_gain = -scorer.score_diff_after_permutation_conjugate(&state, &allowed_perms[i], &cumulative_moves_all[c][p]);
                            let seq_penalty = allowed_sequences[i].len() as i32 - self.simplify_profit3(&sol[..p], &allowed_sequences[i], &sol[p..]) as i32 * 2;

                            if score_gain < 0 || (score_gain == 0 && seq_penalty >= 0) {
                                continue;
                            }
                            let profit = score_gain - seq_penalty * eff;

                            if best_profit < profit {
                                best_profit = profit;
                                best_profit_cands.clear();
                                best_profit_cands.push((total_profit + profit, c, p, i));
                            } else if best_profit == profit {
                                best_profit_cands.push((total_profit + profit, c, p, i));
                            }
                        }
                    }
                }

                best_profit_cands
            }).collect::<Vec<_>>();
            let mut cands_all = vec![];
            for cand in cands {
                cands_all.extend(cand);
            }
            cands_all.shuffle(rng);

            let mut profit_cands_sorter = vec![];
            for i in 0..cands_all.len() {
                // TODO: make this probablistic?
                profit_cands_sorter.push((cands_all[i].0, i));
            }
            profit_cands_sorter.sort();
            profit_cands_sorter.reverse();

            let mut new_candidates = vec![];
            let mut existing_states = HashSet::<Vec<usize>>::new();

            for i in 0..profit_cands_sorter.len() {
                if new_candidates.len() >= beam_width {
                    break;
                }
                let (new_total_profit, cand_id, pos, seq_id) = cands_all[profit_cands_sorter[i].1];
                let cand = &candidates[cand_id];

                let new_state = allowed_perms[seq_id].conjugate(&cumulative_moves_all[cand_id][pos]).apply(&cand.2);
                let new_score = scorer.compute_score(&new_state);

                let mut new_seq: Vec<usize> = vec![];
                let ori_seq = &cand.3;
                new_seq.extend(&ori_seq[..pos]);
                new_seq.extend(&allowed_sequences[seq_id]);
                new_seq.extend(&ori_seq[pos..]);
                let new_seq = self.simplify(&new_seq);

                /*
                let mut state_verify = init_state.clone();
                for &mv in &new_seq {
                    sparse_moves[mv].apply_inplace(&mut state_verify);
                }
                assert_eq!(new_state, state_verify);
                */ 

                if existing_states.contains(&new_state) {
                    continue;
                }
                existing_states.insert(new_state.clone());

                if new_score == 0 {
                    if let Some(cur) = &score_zero_best {
                        if cur.len() > new_seq.len() {
                            score_zero_best = Some(new_seq);
                        }
                    } else {
                        score_zero_best = Some(new_seq);
                    }
                    continue;
                }

                new_candidates.push((new_total_profit, new_score, new_state, new_seq));
            }

            candidates = new_candidates;
        }

        if let Some(score_zero_best) = score_zero_best {
            (score_zero_best, 0)
        } else {
            least_score
        }
    }

    fn solve_local(
        &self,
        mut sol: Vec<usize>,
        scorer: &SimpleScorer,
        state: &mut [usize],
        size_after_restriction: usize,
        dense_moves: &[Permutation],
        combo_sequences: &[Vec<usize>],
        combo_perms: &[SparsePermutation],
        seed: u64)
    -> (Vec<usize>, i32) {
        let mut rng = SmallRng::seed_from_u64(seed);

        let init_len = sol.len();
        loop {
            let score = scorer.compute_score(&state);
            let mut best_move_profit = -100.0f64;
            let mut best_moves = vec![];

            let mut cumulative_moves = vec![Permutation::identity(size_after_restriction)];
            for i in 0..sol.len() {
                cumulative_moves.push(&cumulative_moves[i] * &dense_moves[sol[sol.len() - 1 - i]]);
            }
            cumulative_moves.reverse();

            for p in 0..=sol.len() {
                if 0 < p && p < sol.len() && cumulative_moves[p] == cumulative_moves[p - 1] && cumulative_moves[p] == cumulative_moves[p + 1] {
                    continue;
                }
                for i in 0..combo_perms.len() {
                    let score2 = score + scorer.score_diff_after_permutation_conjugate(&state, &combo_perms[i], &cumulative_moves[p]);

                    let seq_penalty = combo_sequences[i].len() as i32 - self.simplify_profit3(&sol[..p], &combo_sequences[i], &sol[p..]) as i32 * 2;

                    if !(score > score2 || (score == score2 && seq_penalty < 0)) {
                        // no improvement (TODO: allow this in SA)
                        continue;
                    }

                    /*
                    let profit;
                    if seq_penalty < 0 {
                        profit = -10.0 * seq_penalty as f64;
                    } else if seq_penalty == 0 {
                        assert!(score > score2);
                        profit = -10.0 * (score - score2) as f64;
                    } else {
                        profit = (score - score2) as f64 / seq_penalty as f64;
                    }
                    */
                    let profit = (score - score2) as f64 * 1.0 - seq_penalty as f64;
                    if best_move_profit < profit {
                        best_move_profit = profit;
                        best_moves.clear();
                        best_moves.push((p, i, score2));
                    } else if best_move_profit == profit {
                        best_moves.push((p, i, score2));
                    }
                }
            }

            if best_moves.is_empty() {
                break;
            }

            let (pos, id, _s) = best_moves[rng.gen_range(0..best_moves.len())];
            let mut sol2: Vec<usize> = vec![];
            sol2.extend(&sol[..pos]);
            sol2.extend(&combo_sequences[id]);
            sol2.extend(&sol[pos..]);
            // sol = sol2;
            let _len_before = sol.len();
            sol = self.simplify(&sol2);
            // eprintln!("{} -> {} ({:?}, act. cost = {})", score, s, combo_sequences[id], sol.len() - len_before);

            combo_perms[id].conjugate(&cumulative_moves[pos]).apply_inplace(state);
        }
        eprintln!("score: {}, extra {} steps", scorer.compute_score(&state), sol.len() - init_len);

        (sol, scorer.compute_score(&state))
    }

    pub fn resolve_parity(&self, mut sol: Vec<usize>) -> Vec<usize> {
        if self.center_only {
            eprintln!("resolve_parity cannot work on center_only setting");
            return sol;
        }

        let mut reachable_unions = UnionFind::new(self.initial_state.len());
        for mv in &self.sparse_moves {
            for i in 0..mv.num_moves() {
                reachable_unions.join(mv.src(i), mv.dest(i));
            }
        }

        let mut state = self.initial_state.clone();
        for &m in &sol {
            self.sparse_moves[m].apply_inplace(&mut state);
        }

        let mut need_fix_parity = vec![];
        for i in 1..(self.cube_size / 2) {
            let bucket_cells = self.bucket_cells(0, i);
            let mut red = vec![];
            for &c in &bucket_cells {
                if reachable_unions.root(c) == reachable_unions.root(bucket_cells[0]) {
                    red.push(c);
                }
            }

            let mut seq1 = vec![];
            let mut seq2 = vec![];
            for r in red {
                //seq1.push(self.initial_state[r]);
                seq1.push(state[r]);
                seq2.push(self.target_state[r]);
            }

            if !permutation_parity(seq1, seq2) {
                // parity good
                continue;
            }

            eprintln!("parity not good: {i}");
            need_fix_parity.push(i);
        }
        for i in need_fix_parity {
            let mut cumulative_moves = vec![Permutation::identity(self.initial_state.len())];
            for j in 0..sol.len() {
                cumulative_moves.push(&cumulative_moves[j] * &self.dense_moves[sol[sol.len() - 1 - j]]);
            }
            cumulative_moves.reverse();

            let candidate_moves = self.move_ids_for_bucket(i);
            let mut best_move_score = 100000;
            let mut best_move = (0, 0);

            assert!(candidate_moves.len() > 0);

            for p in 0..=sol.len() {
                for &mv in &candidate_moves {
                    let loss = self.scorer.score_diff_after_permutation_conjugate(&state, &self.sparse_moves[mv], &cumulative_moves[p]);
                    if best_move_score > loss {
                        best_move_score = loss;
                        best_move = (p, mv);
                    }
                }
            }

            eprintln!("penalty for fixing parity {}: {}", i, best_move_score);
            let (p, mv) = best_move;
            sol.insert(p, mv);
            self.sparse_moves[mv].conjugate(&cumulative_moves[p]).apply_inplace(&mut state);
        }

        if self.cube_size % 2 == 1 {
            // edge centers have special parity
            let seq1 = get_edge_center_stats(&state, self.cube_size);
            let seq2 = get_edge_center_stats(&self.target_state, self.cube_size);
            let edge_center_parity = permutation_parity(seq1, seq2);

            let seq1 = get_face_center_stats(&state, self.cube_size);
            let seq2 = get_face_center_stats(&self.target_state, self.cube_size);
            let face_center_parity = permutation_parity(seq1, seq2);
            assert!(!face_center_parity, "face_center_parity should have been resolved by center alignment");
            eprintln!("{} {}", edge_center_parity, face_center_parity);

            if edge_center_parity {
                let candidate_moves = self.move_ids_for_combo_bucket(self.cube_size / 2, self.cube_size / 2);
                sol.push(candidate_moves[0]);
            }
        }

        // edge-center nazo parity
        if self.cube_size % 2 == 1 {
            let mut isok = false;

            let candidate_moves = self.move_ids_for_combo_bucket(self.cube_size / 2, self.cube_size / 2);
            let cand_data = self.enumerate_combo();

            for i in 0..37  {
                let mut sol2 = sol.clone();
                if i == 0 {
                    // do nothing
                } else if 1 <= i && i <= 36 {
                    sol2.push(candidate_moves[(i / 6) % 6]);
                    sol2.push(candidate_moves[i % 6]);
                } else {
                    unreachable!();
                }

                if self.solve_local_many(sol2.clone(), &cand_data.edge_center, 0, self.cube_size / 2, 42, 1).1 == 0 {
                    eprintln!("parity resolved pattern {}", i);
                    isok = true;
                    sol = sol2;
                    break;
                }
            }
            if !isok {
                eprintln!("sadparrot");
            }
            // assert!(isok);
        }

        sol
    }

    pub fn solve_local_all(&self, mut sol: Vec<usize>, seed: u64) -> Vec<usize> {
        let cand_data = self.enumerate_combo();
        let bmax = (self.cube_size - 1) / 2;

        let mut last_score = self.evaluate(&sol);
        let mut last_sol_len = sol.len();

        let n_trials = 16;

        if self.center_only {
            for a in 0..=bmax {
                for b in a..=bmax {
                    if self.cube_size % 2 == 1 && a == bmax && b == bmax {
                        continue;
                    }

                    let cand_data = if a == b {
                        &cand_data.diagonal
                    } else if b == self.cube_size / 2 {
                        &cand_data.center_line
                    } else {
                        &cand_data.asymmetry
                    };

                    sol = self.solve_local_many(sol, cand_data, a, b, seed, n_trials).0;
                    // sol = self.solve_local2(sol, cand_data, a, b, 42).0;

                    {
                        let score = self.evaluate(&sol);
                        eprintln!("score: {}; gained {} cells in {} steps", score, last_score - score, sol.len() - last_sol_len);
                        last_score = score;
                        last_sol_len = sol.len();
                    }
                }
            }
        } else {
            for a in 1..=bmax {
                for b in a..=bmax {
                    if self.cube_size % 2 == 1 && a == bmax && b == bmax {
                        continue;
                    }

                    let cand_data = if a == b {
                        &cand_data.diagonal
                    } else if b == self.cube_size / 2 {
                        &cand_data.center_line
                    } else {
                        &cand_data.asymmetry
                    };

                    // sol = self.solve_local_many(sol, &cand_data, a, b, 42, n_trials).0;
                    sol = self.solve_local2(sol, &cand_data, a, b, seed).0;

                    {
                        let score = self.evaluate(&sol);
                        eprintln!("score: {}; gained {} cells in {} steps", score, last_score - score, sol.len() - last_sol_len);
                        last_score = score;
                        last_sol_len = sol.len();
                    }
                }
            }
            for b in 1..=bmax {
                let cand_data = if b == self.cube_size / 2 {
                    &cand_data.edge_center
                } else {
                    &cand_data.edge_asymmetry
                };

                // sol = self.solve_local_many(sol, &cand_data, 0, b, 42, n_trials).0;
                sol = self.solve_local2(sol, &cand_data, 0, b, seed).0;

                {
                    let score = self.evaluate(&sol);
                    eprintln!("score: {}; gained {} cells in {} steps", score, last_score - score, sol.len() - last_sol_len);
                    last_score = score;
                    last_sol_len = sol.len();
                }
            }
        }
        sol
    }

    pub fn evaluate(&self, sol: &[usize]) -> i32 {
        let mut state = self.initial_state.clone();
        for &mv in sol {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }
        self.scorer.compute_score(&state)
    }

    pub fn dump_report(&self, sol: &[usize]) {
        let mut state = self.initial_state.clone();
        for &mv in sol {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }

        let score = self.scorer.compute_score(&state);

        if !self.is_cube {
            // simple report
            eprintln!("steps={}, mismatch={}", sol.len(), score);
            return;
        }

        let n = self.cube_size;

        let mut mismatch_corner = 0;
        let mut mismatch_edge = 0;
        let mut mismatch_center = 0;

        for i in 0..6 {
            for y in 0..n {
                for x in 0..n {
                    let id = i * n * n + y * n + x;
                    if state[id] == self.target_state[id] {
                        continue;
                    }

                    let is_corner = (y == 0 || y == n - 1) && (x == 0 || x == n - 1);
                    let is_edge = (y == 0 || y == n - 1) || (x == 0 || x == n - 1);

                    if is_corner {
                        mismatch_corner += 1;
                    } else if is_edge {
                        mismatch_edge += 1;
                    } else {
                        mismatch_center += 1;
                    }
                }
            }
        }

        eprintln!("steps={}, mismatch={} (corner={}, edge={}, center={})", sol.len(), score, mismatch_corner, mismatch_edge, mismatch_center);
    }

    pub fn show(&self, sol: &[usize]) {
        let mut state = self.initial_state.clone();
        for &mv in sol {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }

        let base36 = |x| {
            if x < 10 {
                (x as u8 + '0' as u8) as char
            } else {
                ((x - 10) as u8 + 'a' as u8) as char
            }
        };
        for i in 0..6 {
            eprintln!("==========================");
            for y in 0..self.cube_size {
                for x in 0..self.cube_size {
                    let id = i * self.cube_size * self.cube_size + y * self.cube_size + x;
                    eprint!("{}", base36(state[id]));
                }
                eprint!(" ");
                for x in 0..self.cube_size {
                    let id = i * self.cube_size * self.cube_size + y * self.cube_size + x;
                    eprint!("{}", base36(self.target_state[id]));
                }
                eprintln!("");
            }
        }
    }

    // Make all fully-center pieces aligned and return additional moves
    pub fn postprocess_center(&self, sol: &[usize]) -> Vec<usize> {
        if self.cube_size % 2 == 0 {
            eprintln!("postprocess_center skipped because cube size is even");
            return vec![];
        }

        assert_eq!(self.cube_size % 2, 1, "works only for odd-sized cubes");
        assert_eq!(self.sparse_moves.len() % 6, 0);

        let mut state = self.initial_state.clone();
        for &id in sol {
            self.sparse_moves[id].apply_inplace(&mut state);
        }

        let move_per_axis = self.sparse_moves.len() / 6;
        assert!(move_per_axis == self.cube_size || move_per_axis == self.cube_size + 2);

        let move_ids = [move_per_axis / 2 * 2, move_per_axis / 2 * 2 + move_per_axis * 2, move_per_axis / 2 * 2 + move_per_axis * 4];
        let mut fully_center_pieces = vec![];
        for i in 0..6 {
            fully_center_pieces.push(i * self.cube_size * self.cube_size + (self.cube_size * self.cube_size / 2));
        }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(([0, 0, 0], state, vec![]));

        while let Some((mvs, state, steps)) = queue.pop_front() {
            let mut isok = true;
            for &p in &fully_center_pieces {
                if state[p] != self.target_state[p] {
                    isok = false;
                }
            }
            let mut diff_cnt = 0;
            for i in 0..3 {
                if mvs[i] != 0 {
                    diff_cnt += 1;
                }
                if mvs[i] == 2 {
                    isok = false;
                }
            }
            if diff_cnt >= 2 {
                isok = false;
            }
            if isok {
                return steps;
            }

            for i in 0..3 {
                {
                    let mut mvs2 = mvs.clone();
                    mvs2[i] = (mvs2[i] + 1) % 4;
                    let state2 = self.sparse_moves[move_ids[i]].apply(&state);
                    let mut steps2 = steps.clone();

                    if !visited.contains(&(mvs2, state2.clone())) {
                        visited.insert((mvs2, state2.clone()));
                        steps2.push(move_ids[i]);
                        queue.push_back((mvs2, state2, steps2));
                    }
                }

                {
                    let mut mvs2 = mvs.clone();
                    mvs2[i] = (mvs2[i] + 3) % 4;
                    let state2 = self.sparse_moves[move_ids[i] ^ 1].apply(&state);
                    let mut steps2 = steps.clone();

                    if !visited.contains(&(mvs2, state2.clone())) {
                        visited.insert((mvs2, state2.clone()));
                        steps2.push(move_ids[i] ^ 1);
                        queue.push_back((mvs2, state2, steps2));
                    }
                }
            }
        }

        panic!();
    }

    pub fn parse_move_string(&self, input: &str) -> Vec<usize> {
        let toks = input.split(".");
        let mut ret = vec![];
        for tok in toks {
            let mut id = None;
            for i in 0..self.move_names.len() {
                if &self.move_names[i] == tok {
                    id = Some(i);
                    break;
                }
            }
            ret.push(id.unwrap());
        }
        ret
    }

    pub fn load_extra_combo(&mut self, combo_path: &str) {
        use std::io::BufRead;
        let file = std::fs::File::open(combo_path).unwrap();
        let lines = std::io::BufReader::new(file).lines().flatten().collect::<Vec<_>>();

        for line in lines {
            if line == "" {
                continue;
            }
            let toks = line.split(",");
            let mut combo = vec![];
            for tok in toks {
                let mut id = None;
                for i in 0..self.move_names.len() {
                    if &self.move_names[i] == tok {
                        id = Some(i);
                    }
                }
                combo.push(id.unwrap());
            }

            self.extra_combo.push(combo);
        }
    }

    pub fn globe_combo(&self) -> Vec<Vec<usize>> {
        let byname = |x: &str| {
            for i in 0..self.move_names.len() {
                if &self.move_names[i] == x {
                    return i;
                }
            }
            panic!();
        };

        let mut rmax = 0;
        let mut fmax = 0;
        for name in &self.move_names {
            if &name[0..1] == "f" {
                fmax = fmax.max(name[1..].parse().unwrap());
            }
            if &name[0..1] == "r" {
                rmax = rmax.max(name[1..].parse().unwrap());
            }
        }
        
        let cands = [byname("r0"), byname("-r0"), byname(&format!("r{}", rmax)), byname(&format!("-r{}", rmax)), byname("f0")];

        let mut combo_candidates = vec![];
        for (idmax, len) in [(15625, 6), (390625, 8)] {
            for id in 0..idmax {

                let mut waf = id;
                let mut seq = vec![];
                for _ in 0..len {
                    let w = waf % 5;
                    waf /= 5;
                    seq.push(w);
                }

                let mut isok = true;
                for i in 0..(len - 1) {
                    let a = seq[i];
                    let b = seq[i + 1];
                    if a < 4 && b < 4 && (a ^ 1) == b {
                        isok = false;
                    }
                    if a >= 4 && b >= 4 && a == b{
                        isok = false;
                    }
                }
                if !isok {
                    continue;
                }

                let mut perm = Permutation::identity(self.initial_state.len());
                for i in 0..len {
                    perm = &self.dense_moves[cands[seq[i]]] * perm;
                }
                let perm = perm.sparsify();
                let n = perm.num_moves();

                if n == 0 {
                    continue;
                }
                if len == 6 && n > 4 {
                    continue;
                }
                if len == 8 && n > 3 {
                    continue;
                }
                /*
                eprintln!("{:?} {}", seq, n);
                for i in 0..seq.len() {
                    eprint!("{} ", nnn[seq[i]]);
                }
                eprintln!();
                for i in 0..n {
                    eprintln!("- {} -> {}", perm.src(i), perm.dest(i));
                }
                */
                combo_candidates.push(seq);
            }
        }

        let mut combo_permutations = HashSet::new();
        let mut unique_combos = vec![];

        for rlo in 0..((rmax + 1) / 2) {
            let rhi = rmax - rlo;
            
            let rlo = byname(&format!("r{}", rlo));
            let rhi = byname(&format!("r{}", rhi));

            for f in 0..=fmax {
                let f = byname(&format!("f{}", f));

                let cands = [rlo, rlo ^ 1, rhi, rhi ^ 1, f];

                for base in &combo_candidates {
                    let mut combo = vec![];
                    for &x in base {
                        combo.push(cands[x]);
                    }

                    let mut mv = Permutation::identity(self.initial_state.len());
                    for &x in &combo {
                        mv = &self.dense_moves[x] * mv;
                    }
                    let mv_sp = mv.sparsify();
                    let n = mv_sp.num_moves();
                    assert!(n == 3 || n == 4, "{} {:?} {:?}", n, base, combo);

                    if combo_permutations.insert(mv) {
                        unique_combos.push(combo);
                    }
                }
            }
        }

        // generate more combo
        if self.use_extended_combo {
            for rlo in 0..((rmax + 1) / 2) {
                let rhi = rmax - rlo;
                
                let rlo = byname(&format!("r{}", rlo));
                let rhi = byname(&format!("r{}", rhi));

                for f in 0..=fmax {
                    let f = byname(&format!("f{}", f));

                    let cands = [rlo, rlo ^ 1, rhi, rhi ^ 1, f];

                    for base in &combo_candidates {
                        if base.len() == 6 {
                            continue;
                        }
                        let mut combo = vec![];
                        for &x in base {
                            combo.push(cands[x]);
                        }

                        let mut mv = Permutation::identity(self.initial_state.len());
                        for &x in &combo {
                            mv = &self.dense_moves[x] * mv;
                        }
                        let mv_sp = mv.sparsify();
                        let n = mv_sp.num_moves();
                        assert!(n == 3);

                        for f2 in 0..=fmax {
                            let f2 = byname(&format!("f{}", f2));
                            let mv2 = &self.dense_moves[f2] * &mv * &self.dense_moves[f2];
                            let mv_sp = mv.sparsify();
                            assert!(mv_sp.num_moves() == 3);

                            if combo_permutations.insert(mv2) {
                                let mut combo2 = vec![f2];
                                combo2.extend(&combo);
                                combo2.push(f2);
                                unique_combos.push(combo2);
                            }
                        }
                    }
                }
            }
        }

        eprintln!("found {} combos", unique_combos.len());
        unique_combos
    }

    fn three_piece(&self, mut seq: Vec<usize>) -> Vec<usize> {
        let combos = self.globe_combo();

        let mut found = HashSet::<SparsePermutation>::new();
        let mut gen_combos = vec![];

        let mut qu = VecDeque::<(SparsePermutation, Vec<usize>)>::new();

        for combo in combos {
            if combo.len() != 8 {
                continue;
            }

            let mut mv = Permutation::identity(self.initial_state.len());
            for i in 0..combo.len() {
                mv = &self.dense_moves[combo[i]] * mv;
            }
            let mv = mv.sparsify();
            if mv.num_moves() != 3 {
                continue;
            }
            let mv = mv.normalize();

            if found.contains(&mv) {
                continue;
            }
            found.insert(mv.clone());
            qu.push_back((mv.clone(), combo.clone()));
            gen_combos.push((mv, combo));
        }

        while let Some((mv, ori)) = qu.pop_front() {
            for i in 0..self.dense_moves.len() {
                let mut combo = vec![i];
                combo.extend(&ori);
                combo.push(i ^ 1);  // gyaku kamo

                let mv = mv.conjugate(&self.dense_moves[i ^ 1]).normalize();
                /*
                {
                    let mut mv3 = Permutation::identity(self.initial_state.len());
                    for j in 0..combo.len() {
                        mv3 = &self.dense_moves[combo[j]] * mv3;
                    }
                    let mv3 = mv3.sparsify().normalize();
                    assert_eq!(mv3.num_moves(), 3);
                    assert_eq!(mv, mv3);
                }
                */

                if found.contains(&mv) {
                    continue;
                }
                found.insert(mv.clone());
                qu.push_back((mv.clone(), combo.clone()));
                gen_combos.push((mv, combo));
            }
        }
        eprintln!("found {} combos", gen_combos.len());

        let mut state = self.initial_state.clone();
        for &mv in &seq {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }
        let mut score = self.scorer.compute_score(&state);
        loop {
            let mut best_update = 1000000;
            let mut best_combo_id = 0;
            // TODO: select shortest one

            for i in 0..gen_combos.len() {
                let upd = self.scorer.score_diff_after_permutation(&state, &gen_combos[i].0);
                if best_update > upd {
                    best_update = upd;
                    best_combo_id = i;
                }
            }
            if best_update >= 0 {
                break;
            }

            seq.extend(&gen_combos[best_combo_id].1);
            gen_combos[best_combo_id].0.apply_inplace(&mut state);
            eprintln!("{} -> {}", score, score + best_update);
            score += best_update;
        }
        if score == 0 {
            eprintln!(":partyparrot:");
        } else {
            eprintln!(":sadparrot:");
        }
        seq
    }

    pub fn wildcard_optimizer(&self, mut seq: Vec<usize>) -> Vec<usize> {
        let mut score = self.evaluate(&seq);

        let mut state = self.initial_state.clone();
        for &mv in &seq {
            self.sparse_moves[mv].apply_inplace(&mut state);
        }

        loop {
            let mut cumulative_moves = vec![Permutation::identity(self.initial_state.len())];
            for i in 0..seq.len() {
                cumulative_moves.push(&cumulative_moves[i] * &self.dense_moves[seq[seq.len() - 1 - i]]);
            }
            cumulative_moves.reverse();

            let mut best_profit = -100000;
            let mut best_opt = None;

            for left in 0..seq.len() {
                let mut mv = Permutation::identity(self.initial_state.len());
                for i in 1..=(seq.len() - left).min(30) {
                    mv = &self.dense_moves[seq[left + i - 1]] * mv;

                    let md = !mv.sparsify();
                    if md.num_moves() > 32 {
                        continue;
                    }
                    let upd = self.scorer.score_diff_after_permutation_conjugate(&state, &md, &cumulative_moves[left + i]);
                    if score + upd > self.wildcard {
                        continue;
                    }

                    let profit = i as i32 - upd * 4;
                    if best_profit < profit {
                        best_profit = profit;
                        best_opt = Some((md, left, i));
                    }
                }
            }

            if let Some((md, left, i)) = best_opt {
                let upd = self.scorer.score_diff_after_permutation_conjugate(&state, &md, &cumulative_moves[left + i]);
                score += upd;
                md.conjugate(&cumulative_moves[left + i]).apply_inplace(&mut state);

                let mut seq2 = vec![];
                seq2.extend(&seq[..left]);
                seq2.extend(&seq[(left + i)..]);
                eprintln!("score: {}, len: {} -> {}", score, seq.len(), seq2.len());
                seq = seq2;
            } else {
                break;
            }
        }

        seq
    }

    pub fn wildcard_optimizer2(&self, seq: Vec<usize>, seed: u64) -> Vec<usize> {
        assert!(!self.center_only);

        let cand_data = self.enumerate_combo();
        let bmax = (self.cube_size - 1) / 2;
        let mut combo_sequences = vec![];
        for a in 0..=bmax {
            for b in a..=bmax {
                if a == 0 && b == 0 {
                    continue;
                }
                if self.cube_size % 2 == 1 && a == bmax && b == bmax {
                    continue;
                }

                let move_ids = self.move_ids_for_combo_bucket(a, b);
                let cand_data = if a == 0 && b == self.cube_size / 2 {
                    &cand_data.edge_center
                } else if a == 0 {
                    &cand_data.edge_asymmetry
                } else if a == b {
                    &cand_data.diagonal
                } else if b == self.cube_size / 2 {
                    &cand_data.center_line
                } else {
                    &cand_data.asymmetry
                };

                for combo in cand_data {
                    let mut ids = vec![];
                    for &i in combo {
                        ids.push(move_ids[i]);
                    }
                    combo_sequences.push(ids);    
                }    
            }
        }

        let mut rng = SmallRng::seed_from_u64(seed);
        self.run_greedy2(seq, &combo_sequences, 2, true, &mut rng)
    }
}
