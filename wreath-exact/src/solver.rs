
use std::{
    collections::{HashMap, VecDeque}, fs::File, io::Read, path::Path, vec
};

use clap::Parser;
use csv::{Error, ReaderBuilder};
use serde::{Deserialize, Deserializer};
use serde;

type Int = u16;
type State = Vec<Int>;
type Moves = HashMap<String, Vec<Int>>;
type HashValue = u128;

#[derive(Deserialize, Debug, Clone)]
struct PuzzleInfo {
    puzzle_type: String, 

    #[serde(deserialize_with = "deserialize_moves_from_str")]
    allowed_moves: Moves,
}

fn deserialize_moves_from_str<'de, D>(deserializer: D) -> Result<Moves, D::Error>
where
    D: Deserializer<'de>,
{
    let v = String::deserialize(deserializer).unwrap();
    let v = v.replace("'", "\"");
    Ok(serde_json::from_str(&v).unwrap())
}

#[derive(Deserialize, Debug, Clone)]
struct SerializedPuzzle {
    id: i32, 
    puzzle_type: String, 
    solution_state: String, 
    initial_state: String, 
    num_wildcards: usize
}

#[derive(Debug, Clone)]
struct Puzzle {
/*     id: i32,  */
    puzzle_type: String, 
    solution_state: State,
    initial_state: State,
    /* num_wildcards: usize,  */
}

fn deserialize_state(str_state: String, _puzzle_type: &String) -> State {
    let mut state = vec![];
    for c in str_state.split(";") {
        let s = match c {
            "A" => 0, 
            "B" => 1, 
            "C" => 2, 
            _ => {
                panic!("Cannot handle puzzle other than wreath")
            }
        };
        state.push(s);
    }
    state
}

fn deserialized_puzzle(puzzle: &SerializedPuzzle) -> Puzzle {
    Puzzle{
/*         id: puzzle.id,  */
        solution_state: deserialize_state(puzzle.solution_state.clone(), &puzzle.puzzle_type),
        initial_state: deserialize_state(puzzle.initial_state.clone(), &puzzle.puzzle_type),
        puzzle_type: puzzle.puzzle_type.clone(),
/*         num_wildcards: puzzle.num_wildcards, */
    }
}


fn augment_reverse_moves(moves: Moves) -> Moves {
    let mut new_moves = HashMap::new();
    for (k, v) in moves.iter() {
        let rev_k = "-".to_string() + k;
        new_moves.insert(k.to_string(), v.clone());

        let mut rev_v = vec![0; v.len()];
        for i in 0..v.len() {
            rev_v[v[i] as usize] = i as Int;
        }
        new_moves.insert(rev_k, rev_v);
    }
    new_moves
}

fn encode_state(state: &State) -> HashValue {
    let mut value = 0;
    for &v in state {
        value = value * 4 + v as u128;
    }
    value
}

fn decode_state(mut state_hash: HashValue, state_size: usize) -> State {
    let mut state = vec![];
    for _ in 0..state_size {
        state.push((state_hash & 3) as Int);
        state_hash >>= 2;
    }
    state.reverse();
    state
}

struct Solver {
    moves: Moves,
}

fn do_move(state: &State, m: &Vec<Int>) -> State {
    let mut new_state = state.clone();
    for i in 0..new_state.len() {
        new_state[i] = state[m[i] as usize];
    }
    new_state
}

impl Solver {
    fn new(moves: Moves) -> Self {
        let moves = augment_reverse_moves(moves);
        dbg!(&moves);
        Solver { moves }
    }

    fn scan_path(
        &mut self,
        src_state: &Vec<u16>,
        dst_state: &Vec<u16>,
        table: &HashMap<HashValue, usize>,
    ) -> Vec<State> {
        let mut path = vec![];
        let mut state = dst_state.clone();
        loop {
            path.push(state.clone());

            if state == src_state.clone() {
                break;
            }

            let mut ok = false;
            let curr_d = *table.get(&encode_state(&state)).unwrap();
            for m in self.moves.values() {
                let prev_state = do_move(&state, m);
                if let Some(&prev_d) = table.get(&encode_state(&prev_state)) {
                    if prev_d + 1 == curr_d {
                        state = prev_state;
                        ok = true;
                        break;
                    }
                }
            }
            assert!(ok);
        }
        path.reverse();
        path
    }

    fn solve(&mut self, src_state: &State, dst_state: &State, initial_moves: String) -> Vec<String> {
        
        let initial_moves: Vec<&str> = initial_moves.trim().split(".").into_iter().filter(|&s| !s.is_empty()).collect();

        let mut tables: [HashMap<HashValue, usize>; 2] = [HashMap::new(), HashMap::new()];
        let mut queues = [VecDeque::new(), VecDeque::new()];
        dbg!(self.moves.len());

        let state_size = src_state.len();
        let mut src_state = src_state.clone();
        for &initial_move in &initial_moves {
            src_state = do_move(&src_state, &self.moves[initial_move])
        }
        dbg!(&src_state);
        let dst_state = dst_state.clone();
        let src_state_code = encode_state(&src_state);
        let dst_state_code = encode_state(&dst_state);
        tables[0].insert(src_state_code, 0);
        queues[0].push_back(src_state_code);
        tables[1].insert(dst_state_code, 0);
        queues[1].push_back(dst_state_code);

        let mut counter = 0;
        let mut middle_state = None;
        // 解は常に1000以下という仮定
        for i in 0..1000 { 
            let (curr, next) = if queues[0].len() <= queues[1].len() {
                (0, 1)
            } else {
                (1, 0)
            };
            
            let mut next_queue = VecDeque::new();
            dbg!(i, queues[curr].len());
            while let Some(state_code) = queues[curr].pop_front() {
                let state = decode_state(state_code, state_size);
                counter += 1;

                let d = *tables[curr]
                    .get(&state_code)
                    .expect("The current state should be already recorded in the table");
                for m in self.moves.values() {
                    let new_state = do_move(&state, m);
                    let new_state_code = encode_state(&new_state);

                    if !tables[curr].contains_key(&new_state_code) {
                        tables[curr].insert(new_state_code, d + 1);
                        next_queue.push_back(new_state_code);
                    }

                    // 双方向探索がぶつかったら、ぶつかった場所を記録して探索を終了する
                    if tables[next].contains_key(&new_state_code) {
                        middle_state = Some(new_state);
                        queues[curr].clear();
                        queues[next].clear();
                        next_queue.clear();
                        break;
                    }
                }
            }
            queues[curr] = next_queue;
            if middle_state.is_some() {
                break;
            }
        }
        eprintln!("num visited: {}", counter);
        let middle_state = middle_state.expect("Failed to find a path");
        let mut forward_path = self.scan_path(&src_state, &middle_state, &tables[0]);
        let mut backward_path = self.scan_path(&dst_state, &middle_state, &tables[1]);

        dbg!(forward_path.len(), backward_path.len());
        backward_path.reverse();
        let backward_path = backward_path[1..].to_vec();
        forward_path.extend(backward_path.clone());


        let mut actions = vec![];
        for &initial_move in &initial_moves {
            actions.push(initial_move.to_string());
        }

        for i in 0..forward_path.len() - 1 {
            let prev_state = forward_path[i].clone();
            let next_state = forward_path[i + 1].clone();
            let mut ok = false;
            for (&ref name, m) in &self.moves {
                if do_move(&prev_state, m) == next_state {
                    actions.push(name.clone());
                    ok = true;
                    break;
                }
            }
            if !ok {
                for m in self.moves.values() {
                    assert_ne!(&do_move(&next_state, m), &prev_state);
                }

                panic!(
                    "Failed to find a action corresponding to state history: prev_state={:?}, next_state={:?}, (index={})",
                    &prev_state, &next_state, i
                );
            }
        }
        assert_eq!(actions.len() + 1, forward_path.len() + initial_moves.len());
        return actions;
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    problem_id: i32,


    #[arg(short, long)]
    data_dir: String,

    #[arg(short, long, default_value_t = String::new())]
    initial_moves: String
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let data_dir = Path::new(&args.data_dir);
    let puzzle_info_path = data_dir.join("puzzle_info.csv");
    let mut puzzle_info_file = File::open(puzzle_info_path).expect("Failed to open puzzle_info_path");
    let mut puzzle_info_data = String::new();
    puzzle_info_file.read_to_string(&mut puzzle_info_data).expect("Failed to read puzzle_info_file");
    let mut reader = ReaderBuilder::new().from_reader(puzzle_info_data.as_bytes());
    let mut puzzle_infos = HashMap::new();
    for result in reader.deserialize() {
        let puzzle_info: PuzzleInfo = result?;
        let puzzle_type = puzzle_info.clone().puzzle_type;
        puzzle_infos.insert(puzzle_type, puzzle_info);
    }

    let puzzle_path = data_dir.join("puzzles.csv");
    let mut puzzle_file = File::open(puzzle_path).expect("Failed to open puzzle_path");
    let mut puzzle_data = String::new();
    puzzle_file.read_to_string(&mut puzzle_data).expect("Failed to read puzzle_file");
    let mut reader = ReaderBuilder::new().from_reader(puzzle_data.as_bytes());

    let mut puzzles = HashMap::new();
    for result in reader.deserialize() {
        let puzzle: SerializedPuzzle = result?;
        let puzzle_id = puzzle.clone().id;
        puzzles.insert(puzzle_id, puzzle);
    }

    let serialized_puzzle = puzzles.get(&args.problem_id).expect("Unknown puzzle_id");
    let puzzle = deserialized_puzzle(serialized_puzzle);
    let moves = puzzle_infos.get(&puzzle.puzzle_type).expect("Unknown puzzle type").allowed_moves.clone();

    let mut solver = Solver::new(moves);
    assert!(
        puzzle.initial_state.len() <= 64,
        "Hash関数のencodeのため一時的に小さいインスタンスのみ許可"
    );
    assert!(
        puzzle.initial_state.iter().max().unwrap() <= &3,
        "Hash関数のencodeのため一時的に小さいインスタンスのみ許可"
    );
    let actions = solver.solve(
        &puzzle.initial_state,
         &puzzle.solution_state, 
        args.initial_moves.clone()
    );

    let mut change_count = 0;
    for i in 1..actions.len() {
        if actions[i] != actions[i - 1] {
            change_count += 1;
        }
    }
    dbg!(change_count);
    dbg!(&actions);
    println!("{}", actions.join("."));
    Ok(())
}
