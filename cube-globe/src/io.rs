use std::fs::File;
use serde_json::Value;
use crate::solver::Puzzle;

struct PuzzleData {
    puzzle_type: String,
    solution_state: Vec<usize>,
    initial_state: Vec<usize>,
    wildcard: i32,
}

fn state_name_to_id(state: &str) -> usize {
    let head = state.as_bytes()[0] - 'A' as u8;

    if head == 'N' as u8 - 'A' as u8 && state.len() > 1 {
        state[1..].parse::<usize>().unwrap()
    } else {
        head as usize
    }
}

fn parse_puzzle_state(state: &str) -> Vec<usize> {
    state.split(";").map(state_name_to_id).collect()
}

fn load_puzzle_data(puzzle_path: &str, target_id: usize) -> PuzzleData {
    let puzzle_file = std::io::BufReader::new(File::open(puzzle_path).unwrap());
    let mut reader = csv::Reader::from_reader(puzzle_file);
    for result in reader.records() {
        let result = result.unwrap();

        let id = result[0].parse::<usize>().unwrap();
        if id != target_id {
            continue;
        }

        let puzzle_type = result[1].to_owned();
        let solution_state = parse_puzzle_state(&result[2]);
        let initial_state = parse_puzzle_state(&result[3]);
        let wildcard = result[4].parse::<i32>().unwrap();

        return PuzzleData {
            puzzle_type,
            solution_state,
            initial_state,
            wildcard,
        }
    }

    panic!();
}

fn load_allowed_moves(puzzle_info_path: &str, target_puzzle_type: &str) -> Vec<(String, Vec<usize>)> {
    let puzzle_file = std::io::BufReader::new(File::open(puzzle_info_path).unwrap());
    let mut reader = csv::Reader::from_reader(puzzle_file);
    for result in reader.records() {
        let result = result.unwrap();

        let puzzle_type = &result[0];
        if puzzle_type != target_puzzle_type {
            continue;
        }

        let allowed_moves_data: Value = serde_json::from_str(&result[1].replace("'", "\"")).unwrap();
        let allowed_moves_data = allowed_moves_data.as_object().unwrap();
        let mut ret = vec![];
        for (key, val) in allowed_moves_data.iter() {
            let val = val.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as usize).collect();
            ret.push((key.clone(), val));
        }
        // TODO: very adhoc
        if target_puzzle_type.starts_with("wreath") {
            return ret;
        }
        ret.sort_by(|(a, _), (b, _)| (&a[0..1], a[1..].parse::<i32>().unwrap()).cmp(&(&b[0..1], b[1..].parse::<i32>().unwrap())));
        return ret;
    }

    panic!();
}

pub fn load(target_id: usize, puzzles_path: &str, puzzle_info_path: &str) -> Puzzle {
    let puzzle = load_puzzle_data(puzzles_path, target_id);
    let puzzle_info = load_allowed_moves(puzzle_info_path, &puzzle.puzzle_type);

    Puzzle::new(puzzle.initial_state, puzzle.solution_state, puzzle_info, puzzle.puzzle_type, puzzle.wildcard)
}
