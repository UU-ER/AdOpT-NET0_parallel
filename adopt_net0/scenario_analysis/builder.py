"""
Writes a page for putting a scenario analysis together by hand.

The catalogue of a case study is a few hundred entries, which is more than
anyone wants to read in a json file, and the definition that goes with it has to
name parameters exactly as the case study spells them. Both problems go away if
the catalogue is something to search rather than to read, so this module writes
it into a page: the parameters on one side, the analysis being built on the
other, and the json it produces underneath, ready to be saved next to the case
study.

The page is one file with the catalogue inside it. That makes it something that
can be opened from a cluster share or sent to a colleague without anything
having to be installed or served, which is the point.

Two things the page does that a json file cannot. Choosing a node or a carrier
is a choice among the ones the case study has, never something to be typed, so a
name that does not exist cannot be written in the first place. And the equations
of a derived quantity are counted as they are typed, so a system that does not
determine what it writes says so while it is being written rather than when it
is checked.

Examples::

    from adopt_net0.scenario_analysis import write_builder

    write_builder("path to input data")
"""

import json
import logging
from pathlib import Path

from .catalogue import scan_input_data

log = logging.getLogger(__name__)

TEMPLATE_FILE = Path(__file__).parent / "builder_template.html"

# The template carries an empty set of data between these two markers, which is
# what keeps it valid javascript on its own. Everything between them, the markers
# included, is replaced by the data of one case study
PLACEHOLDER = "/*__ADOPT_DATA__*/"
END_PLACEHOLDER = "/*__END_DATA__*/"


def write_builder(folder_path: str | Path, save_path: str | Path = None) -> Path:
    """
    Writes the scenario builder of a case study to an html file.

    The catalogue is read from the case study and put inside the page, so the
    file that comes out is self contained and belongs to that case study only.
    Scanning the folder again and writing it again is how the page is kept up to
    date after the case study changes.

    :param str/Path folder_path: path to the folder containing the case study data
    :param str/Path save_path: file the page is written to, by default
        ScenarioBuilder.html inside the input data folder
    :return: path of the page that was written
    :rtype: Path
    """
    folder_path = Path(folder_path)

    with open(folder_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    data = {
        "catalogue": scan_input_data(folder_path),
        "topology": {
            key: topology.get(key, [])
            for key in ["nodes", "carriers", "investment_periods"]
        },
        "case_name": folder_path.name,
    }

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as html_file:
        page = html_file.read()

    start = page.find(PLACEHOLDER)
    end = page.find(END_PLACEHOLDER)
    if start < 0 or end < start:
        raise ValueError(
            f"{TEMPLATE_FILE.name} has no {PLACEHOLDER} ... {END_PLACEHOLDER} for "
            "the data of the case study"
        )

    # A parameter whose name contains the end of a script tag would otherwise
    # close the script the data sits in
    encoded = json.dumps(data, default=str).replace("</", "<\\/")
    page = page[:start] + encoded + page[end + len(END_PLACEHOLDER) :]

    if save_path is None:
        save_path = folder_path / "ScenarioBuilder.html"
    save_path = Path(save_path)

    with open(save_path, "w", encoding="utf-8") as html_file:
        html_file.write(page)

    log.info(
        f"Scenario builder of '{folder_path.name}' written to {save_path}, "
        f"{len(data['catalogue'])} parameters"
    )
    return save_path
