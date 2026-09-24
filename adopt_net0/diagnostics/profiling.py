import cProfile
import csv
import logging
import os
import platform
import re
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psutil

from ..utilities import GUROBI_PARAMETERS

log = logging.getLogger(__name__)

MB = 1024.0 * 1024.0

# Lines of the gurobi log that carry numbers the solver does not expose as a
# model attribute. The root relaxation is the one that matters most: on a MIP
# it runs single threaded and is where a large share of the time goes, so a
# solver option that acts on the root cannot be judged from the total runtime.
#
# "Root relaxation: objective -1.2e+05, 12345 iterations, 3.45 seconds" - the
# objective can also read "cutoff" or "infeasible", so it is skipped over
SOLVER_LOG_PATTERNS = {
    "root_relaxation_iters": re.compile(
        r"^Root relaxation:.*?, (\d+) iterations", re.MULTILINE
    ),
    "root_relaxation_s": re.compile(
        r"^Root relaxation:.*?, \d+ iterations, ([\d.eE+-]+) seconds", re.MULTILINE
    ),
    "presolve_s": re.compile(r"^Presolve time: ([\d.eE+-]+)s", re.MULTILINE),
    "presolved_rows": re.compile(r"^Presolved: (\d+) rows", re.MULTILINE),
    "presolved_cols": re.compile(r"^Presolved: \d+ rows, (\d+) columns", re.MULTILINE),
    "presolved_nnz": re.compile(
        r"^Presolved: \d+ rows, \d+ columns, (\d+) nonzeros", re.MULTILINE
    ),
}

# Header that opens the branch and bound table. Everything before it that looks
# like a row of numbers is the simplex or barrier iteration log, which has the
# same shape and would otherwise be read as node counts
NODE_LOG_HEADER = re.compile(r"^\s*Nodes\s*\|.*Current Node.*\|", re.MULTILINE)

# One row of that table: an optional marker for a heuristic or an improving
# solution, the explored and unexplored node counts, and the elapsed seconds at
# the end of the line
NODE_LOG_ROW = re.compile(r"^[H*]?\s*(\d+)\s+(\d+)\s+.*?(\d+)s\s*$", re.MULTILINE)


class ResourceMonitor:
    """
    Samples resource usage of the current process in a background thread.

    The monitor writes three csv files:

    - profile_timeseries.csv: one row per sample, holding the full resource curve
    - profile_phases.csv: one row per phase, holding average and peak values
    - profile_summary.csv: one row for the whole run, holding model size metrics
      (the predictors) next to the resource metrics (the responses)

    Phases are recorded as time intervals only. Resource values per phase are
    derived from the samples that fall within the interval, which means the
    timeseries can be re-split differently in post-processing.

    If the monitor is disabled, all its methods are no-ops. This allows the
    model to be instrumented without guarding every call.

    :param bool enabled: if False, nothing is sampled and nothing is written
    :param float interval: sampling interval in seconds
    :param bool monitor_cpu: if True, cpu metrics are sampled
    :param bool monitor_memory: if True, memory metrics are sampled
    :param bool monitor_disk: if True, process level disk io is sampled
    :param bool monitor_network: if True, system wide network io is sampled
    """

    def __init__(
        self,
        enabled: bool = True,
        interval: float = 1.0,
        monitor_cpu: bool = True,
        monitor_memory: bool = True,
        monitor_disk: bool = True,
        monitor_network: bool = False,
    ):
        self.enabled = enabled
        self.interval = interval
        self.monitor_cpu = monitor_cpu
        self.monitor_memory = monitor_memory
        self.monitor_disk = monitor_disk
        self.monitor_network = monitor_network

        self.process = psutil.Process(os.getpid())
        self.samples = []
        self.phases = []
        self.metadata = {}

        self._phase_starts = {}
        self._phase_counts = {}
        self._cprofiles = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._t_zero = None

    def start(self):
        """
        Starts the sampling thread
        """
        if not self.enabled or self._thread is not None:
            return

        # Prime cpu_percent, the first call always returns 0.0
        self.process.cpu_percent()

        self._t_zero = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="adopt-resource-monitor", daemon=True
        )
        self._thread.start()

        self._take_sample(event="monitor_start")
        log.info(f"Resource monitor started (interval: {self.interval}s)")

    def stop(self):
        """
        Stops the sampling thread
        """
        if self._thread is None:
            return

        self._take_sample(event="monitor_stop")
        self._stop_event.set()
        self._thread.join(timeout=max(2 * self.interval, 5))
        self._thread = None
        log.info(f"Resource monitor stopped ({len(self.samples)} samples)")

    def phase_start(self, name: str):
        """
        Marks the start of a phase.

        A sample is forced at the boundary, so that even a phase shorter than
        the sampling interval holds at least two samples.

        :param str name: name of the phase
        """
        if not self.enabled:
            return

        with self._lock:
            self._phase_starts.setdefault(name, []).append(time.time())
        self._take_sample(event=f"start:{name}")

    def phase_end(self, name: str):
        """
        Marks the end of a phase and records the resulting interval

        :param str name: name of the phase, as passed to phase_start
        """
        if not self.enabled:
            return

        self._take_sample(event=f"end:{name}")
        t_end = time.time()

        with self._lock:
            starts = self._phase_starts.get(name)
            if not starts:
                log.warning(f"Phase '{name}' was ended without being started")
                return
            t_start = starts.pop()

            # A phase name can occur more than once (pareto, time staging).
            # Repeated occurrences get a numbered suffix.
            count = self._phase_counts.get(name, 0)
            self._phase_counts[name] = count + 1
            phase_name = name if count == 0 else f"{name}_{count + 1}"

            self.phases.append(
                {
                    "phase": phase_name,
                    "t_start": t_start - self._t_zero,
                    "t_end": t_end - self._t_zero,
                    "duration_s": t_end - t_start,
                }
            )

    @contextmanager
    def phase(self, name: str):
        """
        Context manager wrapping phase_start and phase_end.

        A phase named in the environment variable ADOPT_CPROFILE (comma
        separated, e.g. ADOPT_CPROFILE=read_data) is also run under cProfile,
        once with the wall clock and, if ADOPT_CPROFILE_CPU=1, with the cpu
        clock of the process instead. A function whose wall time grows under
        load while its cpu time does not is waiting, one whose cpu time grows
        as well is contending for the cores. The stats are written next to
        the csv files as cprofile_<phase>.prof.

        :param str name: name of the phase
        """
        profiled = self.enabled and name in _cprofile_phases()
        if profiled:
            timer = time.process_time if _cprofile_cpu() else time.perf_counter
            profile = cProfile.Profile(timer)

        self.phase_start(name)
        try:
            if profiled:
                profile.enable()
            yield
        finally:
            if profiled:
                profile.disable()
                self._cprofiles.append((name, profile))
            self.phase_end(name)

    def add_metadata(self, **kwargs):
        """
        Adds key value pairs to the run summary

        :param kwargs: columns to add to profile_summary.csv
        """
        self.metadata.update(kwargs)

    def _run(self):
        """
        Sampling loop of the background thread
        """
        while not self._stop_event.wait(self.interval):
            try:
                self._take_sample()
            except psutil.Error as error:
                log.warning(f"Resource sampling failed: {error}")

    def _take_sample(self, event: str = ""):
        """
        Takes a single sample of all enabled counters

        :param str event: marker written to the event column
        """
        if self._t_zero is None:
            return

        now = time.time()
        sample = {"t": now - self._t_zero, "timestamp": now, "event": event}

        try:
            if self.monitor_memory:
                mem = self.process.memory_info()
                sample["rss_mb"] = mem.rss / MB
                sample["vms_mb"] = mem.vms / MB
                sample["sys_mem_available_mb"] = psutil.virtual_memory().available / MB

            if self.monitor_cpu:
                cpu_times = self.process.cpu_times()
                sample["cpu_percent"] = self.process.cpu_percent()
                sample["cpu_user_s"] = cpu_times.user
                sample["cpu_system_s"] = cpu_times.system
                sample["num_threads"] = self.process.num_threads()

            if self.monitor_disk:
                io = self.process.io_counters()
                sample["io_read_mb"] = io.read_bytes / MB
                sample["io_write_mb"] = io.write_bytes / MB

            if self.monitor_network:
                net = psutil.net_io_counters()
                sample["net_recv_mb"] = net.bytes_recv / MB
                sample["net_sent_mb"] = net.bytes_sent / MB
        except (psutil.Error, OSError) as error:
            log.warning(f"Resource sampling failed: {error}")
            return

        with self._lock:
            self.samples.append(sample)

    def _peak_rss_mb(self):
        """
        Returns the peak rss of the process as reported by the os.

        This is exact, as opposed to the peak of the samples, which can miss
        short spikes. Returns None if the platform does not report it.
        """
        try:
            if platform.system() == "Windows":
                return self.process.memory_info().peak_wset / MB

            import resource

            # ru_maxrss is in kB on linux and in bytes on macOS
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                return max_rss / MB
            return max_rss / 1024.0
        except (AttributeError, ValueError, OSError, ImportError):
            return None

    def _aggregate_phase(self, phase: dict):
        """
        Derives average and peak values for a single phase from the samples

        :param dict phase: phase record holding t_start and t_end
        :return: dict with the aggregated values
        """
        with self._lock:
            samples = [
                s for s in self.samples if phase["t_start"] <= s["t"] <= phase["t_end"]
            ]

        result = dict(phase)
        result["n_samples"] = len(samples)
        if not samples:
            return result

        first, last = samples[0], samples[-1]

        if self.monitor_memory:
            rss = [s["rss_mb"] for s in samples if "rss_mb" in s]
            vms = [s["vms_mb"] for s in samples if "vms_mb" in s]
            if rss:
                result["rss_avg_mb"] = sum(rss) / len(rss)
                result["rss_peak_mb"] = max(rss)
                result["rss_start_mb"] = rss[0]
                result["rss_end_mb"] = rss[-1]
                result["rss_delta_mb"] = rss[-1] - rss[0]
            if vms:
                result["vms_avg_mb"] = sum(vms) / len(vms)
                result["vms_peak_mb"] = max(vms)

        if self.monitor_cpu:
            # cpu_percent of the boundary samples covers time outside the
            # phase, so it is skipped when other samples are available
            inner = samples[1:-1] if len(samples) > 2 else samples
            cpu = [s["cpu_percent"] for s in inner if "cpu_percent" in s]
            if cpu:
                result["cpu_percent_avg"] = sum(cpu) / len(cpu)
                result["cpu_percent_peak"] = max(cpu)
            if "cpu_user_s" in first and "cpu_user_s" in last:
                result["cpu_user_s"] = last["cpu_user_s"] - first["cpu_user_s"]
                result["cpu_system_s"] = last["cpu_system_s"] - first["cpu_system_s"]
                cpu_total = result["cpu_user_s"] + result["cpu_system_s"]
                if phase["duration_s"] > 0:
                    result["parallelism_avg"] = cpu_total / phase["duration_s"]
            threads = [s["num_threads"] for s in samples if "num_threads" in s]
            if threads:
                result["threads_peak"] = max(threads)

        if self.monitor_disk and "io_read_mb" in first and "io_read_mb" in last:
            result["io_read_mb"] = last["io_read_mb"] - first["io_read_mb"]
            result["io_write_mb"] = last["io_write_mb"] - first["io_write_mb"]

        if self.monitor_network and "net_recv_mb" in first:
            result["net_recv_mb"] = last["net_recv_mb"] - first["net_recv_mb"]
            result["net_sent_mb"] = last["net_sent_mb"] - first["net_sent_mb"]

        return result

    def _build_summary(self):
        """
        Builds the single row summary of the run

        :return: dict holding run context, model metrics and resource metrics
        """
        summary = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "n_cpus_physical": psutil.cpu_count(logical=False),
            "n_cpus_logical": psutil.cpu_count(logical=True),
            "sys_mem_total_mb": psutil.virtual_memory().total / MB,
            "sampling_interval_s": self.interval,
            "n_samples": len(self.samples),
        }

        # Scheduler context, empty when not running under slurm
        for column, variable in [
            ("slurm_job_id", "SLURM_JOB_ID"),
            ("slurm_cpus_per_task", "SLURM_CPUS_PER_TASK"),
            ("slurm_mem_per_node", "SLURM_MEM_PER_NODE"),
            ("slurm_nodelist", "SLURM_JOB_NODELIST"),
        ]:
            summary[column] = os.environ.get(variable, "")

        # Threads the numerical libraries were allowed, empty when unset and
        # they took every core. They run the typical day clustering, and left
        # free they multiply the reading time of concurrent runs
        summary["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "")

        summary.update(self.metadata)

        with self._lock:
            samples = list(self.samples)

        if samples:
            summary["wall_total_s"] = samples[-1]["t"]
            rss = [s["rss_mb"] for s in samples if "rss_mb" in s]
            if rss:
                summary["rss_peak_sampled_mb"] = max(rss)
                summary["rss_avg_mb"] = sum(rss) / len(rss)
            if "cpu_user_s" in samples[-1]:
                summary["cpu_user_s"] = samples[-1]["cpu_user_s"]
                summary["cpu_system_s"] = samples[-1]["cpu_system_s"]
            if "io_read_mb" in samples[0] and "io_read_mb" in samples[-1]:
                summary["io_read_mb"] = (
                    samples[-1]["io_read_mb"] - samples[0]["io_read_mb"]
                )
                summary["io_write_mb"] = (
                    samples[-1]["io_write_mb"] - samples[0]["io_write_mb"]
                )

        peak_rss = self._peak_rss_mb()
        if peak_rss is not None:
            summary["rss_peak_os_mb"] = peak_rss

        if samples and "cpu_user_s" in samples[-1] and samples[-1]["t"] > 0:
            cpu_total = samples[-1]["cpu_user_s"] + samples[-1]["cpu_system_s"]
            summary["parallelism_avg"] = cpu_total / samples[-1]["t"]
        if samples:
            threads = [s["num_threads"] for s in samples if "num_threads" in s]
            if threads:
                summary["threads_peak"] = max(threads)

        # Time, cpu and memory of every phase, flattened into columns
        for phase in self.phases:
            aggregated = self._aggregate_phase(phase)
            name = aggregated["phase"]
            summary[f"t_{name}_s"] = aggregated["duration_s"]

            for metric, column in [
                ("rss_peak_mb", f"rss_peak_{name}_mb"),
                ("rss_avg_mb", f"rss_avg_{name}_mb"),
                ("cpu_user_s", f"cpu_user_{name}_s"),
                ("cpu_system_s", f"cpu_system_{name}_s"),
                ("parallelism_avg", f"parallelism_{name}"),
                ("threads_peak", f"threads_peak_{name}"),
            ]:
                if metric in aggregated:
                    summary[column] = aggregated[metric]

        return summary

    def write(self, save_path: Path | str):
        """
        Writes the three csv files to the given folder

        :param Path/str save_path: folder to write the csv files to
        """
        if not self.enabled:
            return

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            samples = list(self.samples)

        if not samples:
            log.warning("No resource samples collected, nothing written")
            return

        phases = [self._aggregate_phase(phase) for phase in self.phases]
        summary = self._build_summary()

        _write_csv(save_path / "profile_timeseries.csv", samples)
        _write_csv(save_path / "profile_phases.csv", phases)
        _write_csv(save_path / "profile_summary.csv", [summary])

        # A phase that occurs more than once gets a numbered suffix, as in
        # profile_phases.csv
        counts = {}
        for name, profile in self._cprofiles:
            counts[name] = counts.get(name, 0) + 1
            suffix = "" if counts[name] == 1 else f"_{counts[name]}"
            profile.dump_stats(str(save_path / f"cprofile_{name}{suffix}.prof"))

        log.info(f"Resource profile written to {save_path}")


def _cprofile_phases():
    """
    Returns the phases to run under cProfile, read from ADOPT_CPROFILE
    """
    value = os.environ.get("ADOPT_CPROFILE", "")
    return {phase.strip() for phase in value.split(",") if phase.strip()}


def _cprofile_cpu():
    """
    Returns True if cProfile is to use the cpu clock, read from ADOPT_CPROFILE_CPU
    """
    return os.environ.get("ADOPT_CPROFILE_CPU", "0") == "1"


def _write_csv(path: Path, rows: list):
    """
    Writes a list of dicts to csv, using the union of all keys as header

    :param Path path: file to write to
    :param list rows: list of dicts
    """
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def collect_profiles(search_path: Path | str, save_path: Path | str = None):
    """
    Collects the run summaries of several runs into a single table.

    All profile_summary.csv files below the search path are read and
    concatenated, resulting in one row per run. This is the table relating the
    size of a case study to the resources it needed. The files that are read
    are left untouched.

    :param Path/str search_path: folder that is searched recursively
    :param Path/str save_path: optional csv file the table is written to
    :return: pandas DataFrame with one row per run
    """
    search_path = Path(search_path)
    summary_files = sorted(search_path.glob("**/profile_summary.csv"))

    if not summary_files:
        log.warning(f"No profile_summary.csv found below {search_path}")
        return pd.DataFrame()

    runs = []
    for summary_file in summary_files:
        run = pd.read_csv(summary_file)
        run["run_folder"] = str(summary_file.parent)
        runs.append(run)

    profiles = pd.concat(runs, ignore_index=True)

    if save_path is not None:
        profiles.to_csv(save_path, index=False)
        log.info(f"Collected {len(profiles)} run profiles to {save_path}")

    return profiles


def load_run_profile(result_folder_path: Path | str):
    """
    Loads the resource curve of a single run and labels it by phase.

    The phase of a sample is derived from the phase intervals, which allows
    the timeseries to be split differently than in profile_phases.csv. Samples
    that fall outside every phase get an empty phase label.

    :param Path/str result_folder_path: results folder of the run
    :return: pandas DataFrame with the samples and a phase column
    """
    result_folder_path = Path(result_folder_path)

    timeseries = pd.read_csv(result_folder_path / "profile_timeseries.csv")
    phases = pd.read_csv(result_folder_path / "profile_phases.csv")

    timeseries["phase"] = ""
    for _, phase in phases.iterrows():
        in_phase = (timeseries["t"] >= phase["t_start"]) & (
            timeseries["t"] <= phase["t_end"]
        )
        timeseries.loc[in_phase, "phase"] = phase["phase"]

    return timeseries


def get_resource_monitor(model_config: dict):
    """
    Initiates the resource monitor and defines its parameters.

    If the configuration does not hold a profiling section, or if profiling is
    switched off, a disabled monitor is returned. All its methods are no-ops.

    :param dict model_config: model configuration
    :return: ResourceMonitor
    """
    profiling_config = model_config.get("profiling", {})

    if not profiling_config.get("profiling_on", {}).get("value", 0):
        return ResourceMonitor(enabled=False)

    return ResourceMonitor(
        enabled=True,
        interval=profiling_config["sampling_interval"]["value"],
        monitor_cpu=bool(profiling_config["monitor_cpu"]["value"]),
        monitor_memory=bool(profiling_config["monitor_memory"]["value"]),
        monitor_disk=bool(profiling_config["monitor_disk"]["value"]),
        monitor_network=bool(profiling_config["monitor_network"]["value"]),
    )


def collect_topology_metrics(data):
    """
    Collects the size metrics of the case study from the data handle.

    These are the predictors that do not require a constructed model.

    :param DataHandle data: data handle of the model
    :return: dict with topology metrics
    """
    metrics = {}

    try:
        topology = data.topology
        periods = topology.get("investment_periods", [])
        nodes = topology.get("nodes", [])

        metrics["n_periods"] = len(periods)
        metrics["n_nodes"] = len(nodes)
        metrics["n_carriers"] = len(topology.get("carriers", []))

        time_index = topology.get("time_index", {})
        for aggregation in ["full", "clustered", "averaged"]:
            if aggregation in time_index:
                metrics[f"n_timesteps_{aggregation}"] = len(time_index[aggregation])

        metrics["resolution_in_h"] = topology.get("resolution_in_h", {}).get("full", "")
        metrics["fraction_of_year_modelled"] = topology.get(
            "fraction_of_year_modelled", ""
        )

        # Technologies, counted over all periods and nodes
        n_technologies = 0
        n_technologies_existing = 0
        for period in data.technology_data:
            for node in data.technology_data[period]:
                for _, tec in data.technology_data[period][node].items():
                    n_technologies += 1
                    if getattr(tec, "existing", 0) == 1:
                        n_technologies_existing += 1
        metrics["n_technologies"] = n_technologies
        metrics["n_technologies_existing"] = n_technologies_existing

        # Networks, counted over all periods
        n_networks = 0
        n_arcs = 0
        for period in data.network_data:
            for _, netw in data.network_data[period].items():
                n_networks += 1
                connection = getattr(netw, "connection", None)
                if connection is not None:
                    n_arcs += int(connection.values.sum())
        metrics["n_networks"] = n_networks
        metrics["n_arcs"] = n_arcs

        config = data.model_config
        metrics["typicaldays_n"] = config["optimization"]["typicaldays"]["N"]["value"]
        metrics["timestaging"] = config["optimization"]["timestaging"]["value"]
        metrics["objective"] = config["optimization"]["objective"]["value"]
        metrics["copperplate"] = config["energybalance"]["copperplate"]["value"]
        metrics["scaling_on"] = config["scaling"]["scaling_on"]["value"]
        metrics["pressure_on"] = config["performance"]["pressure"]["pressure_on"][
            "value"
        ]
        metrics["solver"] = config["solveroptions"]["solver"]["value"]
        metrics["gurobi_threads"] = config["solveroptions"]["threads"]["value"]
        metrics["mipgap"] = config["solveroptions"]["mipgap"]["value"]

        # Every solver option adopt hands to gurobi becomes a column, so that
        # two runs that differ only in one of them can be told apart in the
        # dataset. An option a configuration does not carry was left at the
        # gurobi default, which is what the empty value means
        for option in GUROBI_PARAMETERS:
            if option in ("threads", "mipgap"):
                continue
            metrics[f"gurobi_{option}"] = (
                config["solveroptions"].get(option, {}).get("value", "")
            )
    except (AttributeError, KeyError, TypeError) as error:
        log.warning(f"Could not collect all topology metrics: {error}")

    return metrics


def collect_model_metrics(solver, model=None):
    """
    Collects the size metrics of the optimization problem.

    The numbers are read from the gurobi model, which is free and exact. If it
    is not reachable, the pyomo model is counted instead, which is slower.

    :param solver: pyomo solver object
    :param model: pyomo model, used as fallback
    :return: dict with model size metrics
    """
    metrics = {}

    gurobi_model = getattr(solver, "_solver_model", None)
    if gurobi_model is not None:
        try:
            metrics["n_vars"] = gurobi_model.NumVars
            metrics["n_binvars"] = gurobi_model.NumBinVars
            metrics["n_intvars"] = gurobi_model.NumIntVars
            metrics["n_continuousvars"] = (
                gurobi_model.NumVars - gurobi_model.NumBinVars - gurobi_model.NumIntVars
            )
            metrics["n_constrs"] = gurobi_model.NumConstrs
            metrics["n_qconstrs"] = gurobi_model.NumQConstrs
            metrics["n_sos"] = gurobi_model.NumSOS
            metrics["n_nnz"] = gurobi_model.NumNZs
            metrics["model_density"] = gurobi_model.NumNZs / max(
                gurobi_model.NumVars * gurobi_model.NumConstrs, 1
            )
            return metrics
        except (AttributeError, RuntimeError) as error:
            log.warning(f"Could not read gurobi model metrics: {error}")

    if model is not None:
        try:
            metrics["n_vars"] = model.nvariables()
            metrics["n_constrs"] = model.nconstraints()
        except (AttributeError, RuntimeError) as error:
            log.warning(f"Could not read pyomo model metrics: {error}")

    return metrics


def collect_solver_log_metrics(log_path):
    """
    Reads the numbers gurobi only writes to its log.

    The root relaxation and the presolve are phases of the solve that the
    solver does not expose as attributes, but a solver option that acts on one
    of them cannot be judged from the total runtime. The presolved size is the
    direct measurement of what the presolve did, and unlike a runtime it does
    not move between two runs of the same model.

    :param log_path: path of the solver log, or None
    :return: dict with the metrics found in the log, empty values for the rest
    """
    metrics = {column: "" for column in SOLVER_LOG_PATTERNS}

    if log_path is None:
        return metrics

    log_path = Path(log_path)
    if not log_path.is_file():
        return metrics

    try:
        content = log_path.read_text(errors="replace")
    except OSError as error:
        log.warning(f"Could not read the solver log: {error}")
        return metrics

    for column, pattern in SOLVER_LOG_PATTERNS.items():
        # A log can hold more than one solve, and the model the other metrics
        # are read from is the last one, so the last match is the one to keep
        matches = pattern.findall(content)
        if matches:
            metrics[column] = float(matches[-1])

    metrics["root_node_end_s"] = _root_node_end(content)

    return metrics


def _root_node_end(content: str):
    """
    Elapsed seconds at which the solver left the root node.

    Between the root relaxation and the first branching, the solver sits at the
    root adding cuts and re-solving the LP, and on the models this benchmark
    runs that stretch holds most of the solve. It is reported nowhere as a
    number: it has to be read off the last row of the branch and bound table
    that still shows no explored and no unexplored nodes.

    :param str content: text of the solver log
    :return: float seconds, or "" if the log has no branch and bound table
    """
    header = None
    for header in NODE_LOG_HEADER.finditer(content):
        pass
    if header is None:
        return ""

    at_root = [
        float(seconds)
        for explored, unexplored, seconds in NODE_LOG_ROW.findall(
            content[header.end() :]
        )
        if explored == "0" and unexplored == "0"
    ]

    return at_root[-1] if at_root else ""


def collect_solution_metrics(solver, solution, log_path=None):
    """
    Collects metrics describing how hard the problem was to solve

    :param solver: pyomo solver object
    :param solution: pyomo results object
    :param log_path: path of the solver log, read for the phases of the solve
        that gurobi only reports there
    :return: dict with solution metrics
    """
    metrics = {}

    try:
        metrics["termination_condition"] = str(solution.solver.termination_condition)
        metrics["solver_status"] = str(solution.solver.status)
    except AttributeError:
        pass

    gurobi_model = getattr(solver, "_solver_model", None)
    if gurobi_model is not None:
        for column, attribute in [
            ("gurobi_runtime_s", "Runtime"),
            ("gurobi_work", "Work"),
            ("gurobi_nodecount", "NodeCount"),
            ("gurobi_itercount", "IterCount"),
            ("gurobi_bariter", "BarIterCount"),
            ("gurobi_mipgap", "MIPGap"),
            ("gurobi_objval", "ObjVal"),
            ("gurobi_objbound", "ObjBound"),
        ]:
            try:
                metrics[column] = getattr(gurobi_model, attribute)
            except (AttributeError, RuntimeError):
                metrics[column] = ""

    metrics.update(collect_solver_log_metrics(log_path))

    return metrics
