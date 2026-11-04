import logging
import subprocess
import sys
from pathlib import Path

from .build_analysis import write_builder, write_catalogue
from .define_analysis import read_definition, validate_definition
from .handle_results import collect_results
from .plan_analysis import plan_scenarios, write_plan

log = logging.getLogger(__name__)

SNAKEFILE = Path(__file__).parent / "Snakefile"

# Used where the definition has no resources block
DEFAULT_RESOURCES = {
    "output_folder": "",
    "cores": 4,
    "memory_gb": 8,
    "threads_per_scenario": 1,
    "memory_per_scenario_mb": 1500,
}


class ScenarioHub:
    """
    Class to run a scenario analysis on a case study.

    The analysis is defined in ScenarioAnalysis.json in the input data folder of
    the case study. The hub plans its scenarios, solves them in parallel with
    snakemake and collects the results into one table. Attributes:

    - self.base_path: input data folder of the case study
    - self.definition: the definition of the analysis
    - self.analysis_path: folder the plan, the results and the table go to
    - self.scenarios: the planned scenarios
    """

    def __init__(self):
        """
        Constructor
        """
        self.base_path = None
        self.definition = None
        self.analysis_path = None
        self.scenarios = None

    def read_definition(self, data_path: Path | str):
        """
        Reads the scenario analysis of a case study

        :param Path, str data_path: input data folder of the case study
        """
        self.base_path = Path(data_path)
        self.definition = read_definition(self.base_path / "ScenarioAnalysis.json")

        output_folder = self._resource("output_folder")
        if output_folder:
            output_folder = Path(output_folder)
        else:
            output_folder = self.base_path.parent
        self.analysis_path = (
            output_folder / self.definition["analysis"]["name"]["value"]
        )

    def write_builder(self):
        """
        Writes the parameter catalogue of the case study and the builder page to
        define the analysis with
        """
        write_catalogue(self.base_path)
        page = write_builder(self.base_path)

        log_msg = f"--- Builder page written to {page} ---"
        print(log_msg)
        log.info(log_msg)

    def check(self):
        """
        Checks the definition against the case study
        """
        problems = validate_definition(self.definition, self.base_path)
        if problems:
            raise Exception(
                f"The scenario analysis has {len(problems)} problems:\n- "
                + "\n- ".join(problems)
            )

        log_msg = "--- Scenario analysis checked ---"
        print(log_msg)
        log.info(log_msg)

    def plan(self):
        """
        Checks the definition and writes its scenarios to scenarios.csv
        """
        self.check()
        self.scenarios = plan_scenarios(self.definition)
        write_plan(self.scenarios, self.analysis_path / "scenarios.csv")

        log_msg = (
            f"--- {len(self.scenarios)} scenarios planned in {self.analysis_path} ---"
        )
        print(log_msg)
        log.info(log_msg)

    def run(self, dry_run: bool = False):
        """
        Solves the scenarios that are not solved yet, as many side by side as the
        resources of the definition allow. Run again after a failure or an
        interruption, it continues where it stopped.

        :param bool dry_run: only print what would be run
        """
        if self.scenarios is None:
            self.plan()

        memory_mb = int(self._resource("memory_gb") * 1024)
        command = [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(SNAKEFILE),
            "--config",
            f"base={self.base_path.resolve().as_posix()}",
            f"threads={self._resource('threads_per_scenario')}",
            f"mem_mb={self._resource('memory_per_scenario_mb')}",
            "--cores",
            str(self._resource("cores")),
            "--resources",
            f"mem_mb={memory_mb}",
            # A failed scenario does not stop the others, and one left half done
            # by an interruption is solved again
            "--keep-going",
            "--rerun-incomplete",
            # A scenario is solved when its done record exists, so snakemake's
            # own metadata is not needed
            "--drop-metadata",
        ]
        if dry_run:
            command.append("--dry-run")

        log_msg = (
            f"--- Solving scenarios: {self._resource('cores')} cores, "
            f"{self._resource('memory_gb')} GB ---"
        )
        print(log_msg)
        log.info(log_msg)

        # Run from the analysis folder, so that snakemake works with short
        # relative paths, which Windows needs
        result = subprocess.run(command, cwd=self.analysis_path)
        if result.returncode != 0:
            log_msg = (
                "--- Some scenarios did not finish, see the logs folder. Run again "
                "to retry them ---"
            )
            print(log_msg)
            log.info(log_msg)

    def collect(self):
        """
        Collects the results of all scenarios solved so far into summary.csv

        :return: the table, one row per solved scenario
        :rtype: pd.DataFrame
        """
        table = collect_results(self.analysis_path, self.analysis_path / "summary.csv")

        log_msg = f"--- {len(table)} scenarios collected in summary.csv ---"
        print(log_msg)
        log.info(log_msg)
        return table

    def _resource(self, key: str):
        """
        Reads a value of the resources block of the definition

        :param str key: name of the resource
        :return: its value, or the default if the definition does not give it
        """
        resources = self.definition.get("resources", {})
        return resources.get(key, {}).get("value", DEFAULT_RESOURCES[key])
