import json
import logging
from pathlib import Path

import pandas as pd

from .plan_analysis import read_plan

log = logging.getLogger(__name__)

# Columns of adopt's summary that are bookkeeping rather than results
SKIPPED_SUMMARY = ["case", "time_stamp", "pareto_point", "time_stage", "objective"]


def collect_results(analysis_path: str | Path, save_path: str | Path = None):
    """
    Collects the results of every scenario solved so far into one table: name,
    label, values, how the solve ended and the summary adopt wrote. Scenarios
    not solved yet are left out

    :param str/Path analysis_path: folder of the analysis, holding scenarios.csv,
        done/ and results/
    :param str/Path save_path: csv file the table is written to, if given
    :return: the table, one row per solved scenario
    :rtype: pd.DataFrame
    """
    analysis_path = Path(analysis_path)
    scenarios = read_plan(analysis_path / "scenarios.csv")

    rows = []
    for scenario in scenarios:
        done_file = analysis_path / "done" / f"{scenario['name']}.json"
        if not done_file.exists():
            continue

        with open(done_file, "r") as json_file:
            record = json.load(json_file)

        row = {"name": scenario["name"], "label": scenario["label"]}
        row.update(scenario["values"])
        row.update({key: record[key] for key in ["termination", "wall_s", "threads"]})
        row.update(_summary(analysis_path / "results" / scenario["name"]))
        rows.append(row)

    table = pd.DataFrame(rows)
    log.info(f"{len(rows)} of {len(scenarios)} scenarios solved")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, index=False)

    return table


def _summary(results_path: Path) -> dict:
    """
    Reads the last row of the summary adopt wrote for one scenario (a retried
    scenario has one row per attempt)

    :param Path results_path: results folder of the scenario
    :return: dict of the summary, empty if there is none
    """
    summary_file = results_path / "Summary.xlsx"
    if not summary_file.exists():
        return {}

    summary = pd.read_excel(summary_file).iloc[-1]
    return {key: value for key, value in summary.items() if key not in SKIPPED_SUMMARY}
