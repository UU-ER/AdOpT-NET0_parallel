"""
Measures how much each complexity knob adds to the size of the problem.

The model is built and translated into the gurobi model, but not solved, which
makes it cheap enough to sweep all knobs. Starting from a baseline where every
knob is off, each knob is switched on one at a time.

Example::

    python measure_complexity.py --case four_node --typicaldays 2
"""

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import adopt_net0 as adopt

from case_studies import CASE_STUDIES

BASE = Path(__file__).parent
INPUT_DATA_PATH = BASE / "inputData"
RESULTS_PATH = BASE / "results"

# Baseline: every knob at its cheapest setting
BASELINE = {
    "typicaldays_method": 1,
    "electricity_price": "constant",
    "hydrogen_demand": "constant",
    "pipeline_capex": "linear",
    "bidirectional_precise": 0,
    "pipeline_size_min": 0,
    "storage": "on",
    "storage_precise": 0,
    "pv": "on",
}

# One knob switched on at a time
KNOBS = {
    "typicaldays_method": 2,
    "electricity_price": "fluctuating",
    "hydrogen_demand": "fluctuating",
    "pipeline_capex": "fixed_plus_linear",
    "bidirectional_precise": 1,
    "pipeline_size_min": 250,
    "storage": "off",
    "storage_precise": 1,
    "pv": "off",
}


def measure(case: str, typicaldays: int, settings: dict):
    """
    Builds the model and reads its size without solving it

    :param str case: name of the case study
    :param int typicaldays: number of typical days
    :param dict settings: knob settings
    :return: dict with the size of the problem
    """
    case_study = CASE_STUDIES[case]
    input_data_path = INPUT_DATA_PATH / case

    case_study.setup(
        input_data_path,
        RESULTS_PATH,
        typicaldays=typicaldays,
        solver="gurobi_persistent",
        case_name="complexity_measurement",
        **settings,
    )

    model = adopt.ModelHub()
    model.read_data(input_data_path)
    model.construct_model()
    model.construct_balances()

    # For a persistent solver, _define_solver_settings calls set_instance,
    # which translates the pyomo model into the gurobi model. That is all that
    # is needed to read the size, so the model is never optimized and no
    # objective is set.
    model._define_solver_settings()

    # Gurobi only populates the size attributes after an update
    gurobi_model = model.solver._solver_model
    gurobi_model.update()

    size = {
        "n_vars": gurobi_model.NumVars,
        "n_binvars": gurobi_model.NumBinVars,
        "n_intvars": gurobi_model.NumIntVars,
        "n_constrs": gurobi_model.NumConstrs,
        "n_nnz": gurobi_model.NumNZs,
    }

    # A measurement can leave several GB behind, which slows down the next one
    # to the point where its construction time is meaningless. The model is
    # therefore released explicitly before returning.
    gurobi_model.dispose()
    del gurobi_model
    del model
    gc.collect()

    return size


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="four_node", choices=list(CASE_STUDIES))
    parser.add_argument("--typicaldays", type=int, default=2)
    parser.add_argument(
        "--knobs",
        nargs="+",
        choices=list(KNOBS),
        help="knobs to measure, default all of them",
    )
    args = parser.parse_args()

    knobs = {k: v for k, v in KNOBS.items() if not args.knobs or k in args.knobs}

    results = {}

    print(f"\nBaseline (all knobs off), {args.typicaldays} typical days")
    baseline = measure(args.case, args.typicaldays, dict(BASELINE))
    results["baseline"] = baseline

    for knob, value in knobs.items():
        settings = dict(BASELINE)
        settings[knob] = value
        print(f"\nKnob: {knob} = {value}")
        results[f"{knob}={value}"] = measure(args.case, args.typicaldays, settings)

    print("\n" + "=" * 96)
    print(
        f"{'configuration':38s} {'vars':>9} {'binary':>8} {'integer':>8} "
        f"{'constrs':>9} {'nonzeros':>10}"
    )
    print("=" * 96)
    for name, size in results.items():
        print(
            f"{name:38s} {size['n_vars']:9d} {size['n_binvars']:8d} "
            f"{size['n_intvars']:8d} {size['n_constrs']:9d} {size['n_nnz']:10d}"
        )

    print("\nDifference to the baseline")
    for name, size in results.items():
        if name == "baseline":
            continue
        print(
            f"{name:38s} "
            f"{size['n_vars'] - baseline['n_vars']:+9d} "
            f"{size['n_binvars'] - baseline['n_binvars']:+8d} "
            f"{size['n_intvars'] - baseline['n_intvars']:+8d} "
            f"{size['n_constrs'] - baseline['n_constrs']:+9d} "
            f"{size['n_nnz'] - baseline['n_nnz']:+10d}"
        )


if __name__ == "__main__":
    main()
