"""
RTSP Benchmark Instance Management Tool – Unified Entry Point
=============================================================
Usage:
    python main.py generate  [options]
    python main.py validate  --instance <file> --solution <file>
    python main.py solve     --solver <name> --instance <file> [solver options]
    python main.py list-datasets
    python main.py list-solvers

Run `python main.py <command> --help` for per-command help.
"""

import argparse
import sys
import os


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------
def cmd_generate(args):
    """Delegate to generator.py logic (imported as a module-level call)."""
    # Build the argv that generator.py expects and re-invoke it via subprocess
    # so it can use its own argparse without conflict.
    import subprocess
    cmd = [sys.executable, 'generator.py']
    if args.params:
        cmd += ['--params', args.params]
    if args.gen_params:
        cmd += ['--gen-params', args.gen_params]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Sub-command: validate
# ---------------------------------------------------------------------------
def cmd_validate(args):
    from validator import main as validate_main
    validate_main(args.instance, args.solution)


# ---------------------------------------------------------------------------
# Sub-command: solve
# ---------------------------------------------------------------------------
def cmd_solve(args):
    solver = args.solver.lower().replace('-', '_').replace(' ', '_')

    if solver in ('heuristic', 'heuristics', 'first_fit'):
        _solve_heuristic(args)
    elif solver in ('sa', 'simulated_annealing'):
        _solve_sa(args)
    elif solver in ('sa_v2', 'simulated_annealing_v2'):
        _solve_sa_v2(args)
    elif solver in ('milp', 'cplex', 'cplex_milp'):
        _solve_milp(args)
    else:
        print(f"[ERROR] Unknown solver '{args.solver}'. Run `python main.py list-solvers` to see available options.")
        sys.exit(1)


def _solve_heuristic(args):
    import json
    from data_structure.Input import Input
    from solvers.heuristics.heuristics import heuristicFirstFit

    with open('generator_parameter.json') as f:
        parameters = json.load(f)

    obj = args.obj if hasattr(args, 'obj') and args.obj is not None else 1
    objWeights = [w for w in parameters['weightsAlpha'][obj].values()]

    inputData = Input(args.instance)
    sa, _, elapsed = heuristicFirstFit(inputData, objWeights)

    print(f"Heuristic (FirstFit) completed in {elapsed:.2f}s")
    if sa.solution.checkIfLegal(inputData):
        print("Solution is FEASIBLE.")
        print('objective is: ', sa.fitness.objective)
        _save_solution(sa, args)
    else:
        print("Solution is NOT feasible.")


def _solve_sa(args):
    import copy
    import json
    from data_structure.Input import Input
    from solvers.heuristics.heuristics import heuristicFirstFit
    from solvers.simulated_annealing.SimulatedAnnealing import SimulatedAnnealing

    with open('generator_parameter.json') as f:
        parameters = json.load(f)

    obj = args.obj if hasattr(args, 'obj') and args.obj is not None else 1
    objWeights = [w for w in parameters['weightsAlpha'][f'{obj}'].values()]

    inputData = Input(args.instance)
    neighbour_sizes = _parse_neighbour_sizes(args)
    neighbour_prob = _parse_neighbour_prob(args)

    if args.solution:
        sa = SimulatedAnnealing(
            args.instance,
            args.solution,
            objWeights,
            neighbour_sizes,
            capacityPenaltyWeight=args.capacity_penalty_weight,
        )
    else:
        initial_solution, _, _ = heuristicFirstFit(copy.deepcopy(inputData))
        sa = SimulatedAnnealing(
            args.instance,
            None,
            objWeights,
            neighbour_sizes,
            inputData=inputData,
            data={
                'patientAppointments': initial_solution.patientAppointments,
                'startDays': initial_solution.startDays,
            },
            capacityPenaltyWeight=args.capacity_penalty_weight,
        )

    sa.run(
        start_temperature=args.start_temperature,
        k_max=args.k_max,
        cooling_rate=args.cooling_rate,
        neighbour_prob=neighbour_prob,
        seed=args.seed,
        max_iter=args.max_iter,
    )

    print("Simulated Annealing completed.")
    if sa.solution.checkIfLegal(sa.input):
        print("Solution is FEASIBLE.")
        print('objective is: ', sa.fitness.objective)
        _save_solution(sa, args)
    else:
        print("Solution is NOT feasible.")


def _solve_sa_v2(args):
    import copy
    import json
    from data_structure.Input import Input
    from solvers.heuristics.heuristics import heuristicFirstFit
    from solvers.simulated_annealing.SimulatedAnnealingV2 import SimulatedAnnealingV2

    with open('generator_parameter.json') as f:
        parameters = json.load(f)

    obj = args.obj if hasattr(args, 'obj') and args.obj is not None else 1
    objWeights = [w for w in parameters['weightsAlpha'][f'{obj}'].values()]

    inputData = Input(args.instance)
    neighbour_sizes = _parse_neighbour_sizes(args)
    neighbour_prob = _parse_neighbour_prob(args)

    if args.solution:
        sa = SimulatedAnnealingV2(
            args.instance,
            args.solution,
            objWeights,
            neighbour_sizes,
            capacityPenaltyWeight=args.capacity_penalty_weight,
        )
    else:
        initial_solution, _, _ = heuristicFirstFit(copy.deepcopy(inputData))
        sa = SimulatedAnnealingV2(
            args.instance,
            None,
            objWeights,
            neighbour_sizes,
            inputData=inputData,
            data={
                'patientAppointments': initial_solution.patientAppointments,
                'startDays': initial_solution.startDays,
            },
            capacityPenaltyWeight=args.capacity_penalty_weight,
        )

    sa.run(
        start_temperature=args.start_temperature,
        final_temperature=args.final_temperature,
        cooling_rate=args.cooling_rate,
        neighbour_prob=neighbour_prob,
        seed=args.seed,
        total_iterations=args.iterations_total,
        sigma=args.sigma,
        max_iter=args.max_iter,
    )

    print("Simulated Annealing V2 completed.")
    if sa.solution.checkIfLegal(sa.input):
        print("Solution is FEASIBLE.")
        _save_solution(sa, args)
    else:
        print("Solution is NOT feasible.")


def _solve_milp(args):
    import json
    from data_structure.Input import Input
    from solvers.cplex_milp.batch_milp import solve as milp_solve

    with open('generator_parameter.json') as f:
        parameters = json.load(f)

    obj = args.obj if hasattr(args, 'obj') and args.obj is not None else 1
    objWeights = [w for w in parameters['weightsAlpha'][obj].values()]

    inputData = Input(args.instance)
    result = milp_solve(inputData, objWeights)
    print("MILP solver completed.")
    print(result)


def _save_solution(sa, args):
    import pickle
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.instance))[0]
        out_path = f'solutions/{base}_solution.pk'

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    sol_to_store = {
        'patientAppointments': sa.solution.patientAppointments,
        'startDays': sa.solution.startDays,
        'acceptedMovesCounter': sa.acceptedMovesCounter,
        'improvingMovesCounter': sa.improvingMovesCounter,
        'acceptedMovesGain': sa.acceptedMovesGain,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(sol_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Solution saved to: {out_path}")


def _parse_neighbour_sizes(args):
    if hasattr(args, 'neighbour_sizes') and args.neighbour_sizes:
        values = [int(x) for x in args.neighbour_sizes.split(',')]
    else:
        values = [5, 5, 5, 5]

    if len(values) != 4:
        raise ValueError("Expected four neighbour sizes: twDays,twShift,machineDays,machineShift.")

    return values


def _parse_neighbour_prob(args):
    if hasattr(args, 'neighbour_prob') and args.neighbour_prob:
        values = [float(x) for x in args.neighbour_prob.split(',')]
    else:
        values = [0.2, 0.2, 0.2, 0.2, 0.2]

    if len(values) != 5:
        raise ValueError("Expected five neighbourhood probabilities.")

    total = sum(values)
    if total <= 0:
        raise ValueError("Neighbourhood probabilities must sum to a positive value.")

    return [value / total for value in values]


# ---------------------------------------------------------------------------
# Sub-command: list-datasets
# ---------------------------------------------------------------------------
def cmd_list_datasets(args):
    import glob, json
    configs = sorted(glob.glob('gen_input_params/*.json'))
    if not configs:
        print("No config files found in gen_input_params/.")
        return
    print(f"{'Config file':<35} {'Dataset title':<25} {'Machines':>8} {'TW':>4} {'Instances':>10} {'Arrival mean':>13}")
    print('-' * 100)
    for path in configs:
        with open(path) as f:
            cfg = json.load(f)
        print(
            f"{os.path.basename(path):<35} "
            f"{cfg.get('dataset_title',''):<25} "
            f"{cfg.get('machines_number','?'):>8} "
            f"{cfg.get('time_windows_number','?'):>4} "
            f"{cfg.get('instances_number','?'):>10} "
            f"{cfg.get('arrival_mean','?'):>13}"
        )


# ---------------------------------------------------------------------------
# Sub-command: list-solvers
# ---------------------------------------------------------------------------
def cmd_list_solvers(_args):
    solvers = [
        ('heuristic', 'Greedy FirstFit heuristic (fast, good initial solution)'),
        ('sa',        'Simulated Annealing metaheuristic (slower, better quality)'),
        ('sa_v2',     'Simulated Annealing with derived N_s/N_a and cooling until the final temperature'),
        ('milp',      'CPLEX MILP exact solver (requires IBM CPLEX license)'),
    ]
    print("Available solvers (use with `python main.py solve --solver <name>`):\n")
    for name, desc in solvers:
        print(f"  {name:<12}  {desc}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='RTSP Benchmark Instance Management Tool – unified entry point.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', metavar='<command>')
    sub.required = True

    # generate
    p_gen = sub.add_parser('generate', help='Generate benchmark instances from JSON config files.')
    p_gen.add_argument(
        '-p', '--params', default=None, metavar='PATTERN',
        help='Glob pattern inside gen_input_params/ (e.g. "dataset_1_7_2.json"). Default: all files.'
    )
    p_gen.add_argument(
        '--gen-params', default='generator_parameter.json', metavar='FILE',
        help='Path to global generator_parameter.json (default: generator_parameter.json).'
    )
    p_gen.set_defaults(func=cmd_generate)

    # validate
    p_val = sub.add_parser('validate', help='Check feasibility of a solution file.')
    p_val.add_argument('--instance', required=True, metavar='FILE', help='Path to the instance .pk file.')
    p_val.add_argument('--solution', required=True, metavar='FILE', help='Path to the solution .pk file.')
    p_val.set_defaults(func=cmd_validate)

    # solve
    p_sol = sub.add_parser('solve', help='Run a solver on an instance.')
    p_sol.add_argument(
        '--solver', required=True, metavar='NAME',
        help='Solver to use: heuristic | sa | milp. Run `list-solvers` for details.'
    )
    p_sol.add_argument('--instance', required=True, metavar='FILE', help='Path to the instance .pk file.')
    p_sol.add_argument('--solution', default=None, metavar='FILE',
                       help='(SA only) Path to an initial solution .pk file.')
    p_sol.add_argument('--output', default=None, metavar='FILE',
                       help='Where to save the output solution (default: solutions/<instance>_solution.pk).')
    p_sol.add_argument('--obj', type=int, default=1, metavar='N',
                       help='Objective weights index in generator_parameter.json (default: 1).')
    p_sol.add_argument('--neighbour-sizes', default='5,5,5,5', metavar='N,N,N,N',
                       help='(SA only) Comma-separated neighbour sizes (default: 5,5,5,5).')
    p_sol.add_argument('--neighbour-prob', default='0.2,0.2,0.2,0.2,0.2', metavar='P,P,P,P,P',
                       help='(SA only) Comma-separated neighbourhood probabilities (default: uniform).')
    p_sol.add_argument('--start-temperature', type=float, default=100.0, metavar='T',
                       help='(SA only) Initial temperature (default: 100.0).')
    p_sol.add_argument('--final-temperature', type=float, default=1.0, metavar='T',
                       help='(SA V2 only) Final temperature threshold T_f (default: 1.0).')
    p_sol.add_argument('--cooling-rate', type=float, default=0.99, metavar='A',
                       help='(SA only) Multiplicative cooling rate alpha (default: 0.99).')
    p_sol.add_argument('--k-max', type=int, default=1000, metavar='N',
                       help='(SA only) Total number of neighbourhood moves (default: 1000).')
    p_sol.add_argument('--seed', type=int, default=42, metavar='N',
                       help='(SA only) Random seed (default: 42).')
    p_sol.add_argument('--max-iter', type=int, default=50, metavar='N',
                       help='(SA only) Max retries when building a neighbourhood move (default: 50).')
    p_sol.add_argument('--capacity-penalty-weight', type=float, default=100.0, metavar='W',
                       help='(SA only) Weight used for capacity violations in the fitness penalty (default: 100).')
    p_sol.add_argument('--iterations-total', type=int, default=1000, metavar='I',
                       help='(SA V2 only) Total iteration budget I used to derive N_s (default: 1000).')
    p_sol.add_argument('--sigma', type=float, default=0.2, metavar='S',
                       help='(SA V2 only) Multiplier used to derive N_a = sigma * N_s (default: 0.2).')
    p_sol.set_defaults(func=cmd_solve)

    # list-datasets
    p_ld = sub.add_parser('list-datasets', help='List all available dataset config files.')
    p_ld.set_defaults(func=cmd_list_datasets)

    # list-solvers
    p_ls = sub.add_parser('list-solvers', help='List available solvers.')
    p_ls.set_defaults(func=cmd_list_solvers)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
