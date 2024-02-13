fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut opts = getopts::Options::new();
    opts.optopt("", "id", "puzzle id", "ID");
    opts.optopt("", "puzzle-info-path", "path to puzzle_info.csv", "PUZZLE_INFO");
    opts.optopt("", "puzzles-path", "path to puzzles.csv", "PUZZLES");
    opts.optflag("", "center-only", "if set, consider only center pieces");
    opts.optopt("", "extra-combo", "extra combo file", "EXTRA_COMBO");
    opts.optopt("", "np", "thread parallel", "NUM_THREADS");
    opts.optopt("", "seed", "random seed", "SEED");
    opts.optopt("", "seed2", "random seed2", "SEED2");
    opts.optopt("", "thr", "reduction threshold cube", "THR");
    opts.optopt("", "perturb", "random perturbation", "PERTURBATION_LEVEL");
    opts.optflag("", "use-extended-combo", "use extended combo");
    opts.optopt("", "optimize-for-wildcard", "run wildcard optimizer", "MOVES");

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(e) => { panic!("{}", e.to_string()) },
    };

    let puzzle_id = matches.opt_str("id").expect("--id must be specified").parse::<usize>().expect("--id could not be parsed");
    let puzzle_info_path = matches.opt_str("puzzle-info-path").expect("--puzzle-info-path is required");
    let puzzles_path = matches.opt_str("puzzles-path").expect("--puzzles-path is required");
    let extra_combo = matches.opt_str("extra-combo");
    let center_only = matches.opt_present("center-only");
    let _num_threads = matches.opt_str("np").unwrap_or(String::from("16")).parse::<i32>().unwrap();
    let seed = matches.opt_str("seed").unwrap_or(String::from("42")).parse::<u64>().unwrap();
    let seed2 = matches.opt_str("seed2").unwrap_or(String::from("49")).parse::<u64>().unwrap();
    let thr = matches.opt_str("thr").unwrap_or(String::from("4")).parse::<i32>().unwrap();
    let perturb = matches.opt_str("perturb").map(|x| x.parse::<usize>().unwrap());

    let puzzle = beam::io::load(puzzle_id, &puzzles_path, &puzzle_info_path);
    let is_cube = puzzle.is_cube();
    let optimize_for_wildcard = matches.opt_str("optimize-for-wildcard");

    let mut solver = beam::solver::Solver::new(puzzle, center_only, true, optimize_for_wildcard.is_some());

    if let Some(initial_move) = optimize_for_wildcard {
        let moves = solver.parse_move_string(&initial_move);
        let ans = solver.wildcard_optimizer2(moves, seed);

        println!("{}", solver.solution_to_string(&ans));
        solver.dump_report(&ans);
        return;
    }

    if let Some(extra_combo) = extra_combo {
        eprintln!("=== load extra combo ===");
        solver.load_extra_combo(&extra_combo);
    }

    if is_cube {
        eprintln!("=== initial greedy ===");
        let mut ans = solver.greedy2(seed, perturb, thr);
        eprintln!("score after greedy: {}", solver.evaluate(&ans));

        eprintln!("=== center alignment ===");
        let postproc = solver.postprocess_center(&ans);
        eprintln!("postproc len: {}", postproc.len());
        ans.extend(&postproc);
        eprintln!("score after center alignment: {}", solver.evaluate(&ans));

        eprintln!("=== resolve parity ===");
        let ans = solver.resolve_parity(ans.clone());
        // eprintln!("=== resolve parity again ===");
        // solver.resolve_parity(ans.clone());
        eprintln!("score after parity resolution: {}", solver.evaluate(&ans));

        // debug
        let a = ans.len();
        let b = solver.evaluate(&ans);

        eprintln!("=== 24/48 faces problem ===");
        let ans = solver.solve_local_all(ans, seed2);

        println!("{}", solver.solution_to_string(&ans));

        eprintln!("before 48 faces problem: steps={}, mismatch={}", a, b);
        solver.dump_report(&ans);
    } else {
        eprintln!("=== initial greedy ===");
        if matches.opt_present("use-extended-combo") {
            solver.set_use_extended_combo();
        }
        let ans = solver.greedy2(seed, perturb, thr);

        println!("{}", solver.solution_to_string(&ans));

        solver.dump_report(&ans);
    }
}
