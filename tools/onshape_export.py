"""
Run onshape-to-robot on a robot directory, optionally renaming mates first.

onshape-to-robot only turns mates named "dof_*" into joints. Some team
assemblies (e.g. the arm) have correct Revolute mates with default names, and
we must not edit the team's documents. This wrapper renames mates in the
downloaded data only, never in Onshape, using an optional "rename_mates" map
in the robot's config.json:

    "rename_mates": {"Revolute 1": "dof_arm_2", ...}

Usage (from the repo root):
    source .env.onshape
    .venv-onshape/bin/python tools/onshape_export.py export/arm [--save-pickle]
"""

import json
import sys
import time
from pathlib import Path

import requests
from onshape_to_robot import assembly
from onshape_to_robot.export import main
from onshape_to_robot.onshape_api import onshape

# Onshape resets the connection when requests come in bursts. Space requests
# out and retry resets with growing waits; finished downloads are cached by
# onshape-to-robot, so a retry never repeats earlier calls.
REQUEST_SPACING_S = 0.5
RETRY_WAITS_S = [10, 30, 60, 120]
original_request = onshape.Onshape.request


def throttled_request(self, *args, **kwargs):
    for wait in [*RETRY_WAITS_S, None]:
        time.sleep(REQUEST_SPACING_S)
        try:
            return original_request(self, *args, **kwargs)
        except requests.exceptions.ConnectionError:
            if wait is None:
                raise
            print(f"Onshape reset the connection, waiting {wait}s before retrying")
            time.sleep(wait)


onshape.Onshape.request = throttled_request

robot_dir = Path(sys.argv[1])
robot_config = json.loads((robot_dir / "config.json").read_text())
renames = robot_config.get("rename_mates", {})
# onshape-to-robot roots the kinematic tree at the first top-level instance;
# "root_instance" (e.g. "Part 1 <1>") moves that instance first.
root_instance = robot_config.get("root_instance")

original_retrieve_assembly = assembly.Assembly.retrieve_assembly
original_load_features = assembly.Assembly.load_features


def retrieve_assembly(self):
    original_retrieve_assembly(self)
    instances = self.assembly_data["rootAssembly"]["instances"]
    if root_instance:
        instances.sort(key=lambda instance: instance["name"] != root_instance)
    for feature in self.assembly_data["rootAssembly"]["features"]:
        data = feature["featureData"]
        if data.get("name") in renames:
            data["name"] = renames[data["name"]]


def load_features(self):
    original_load_features(self)
    # Joint limits and offsets are looked up by mate name, so rename here too
    for feature in self.features["features"]:
        message = feature["message"]
        if message.get("name") in renames:
            message["name"] = renames[message["name"]]
    for entry in (self.matevalues or {}).get("mateValues", []):
        if entry.get("mateName") in renames:
            entry["mateName"] = renames[entry["mateName"]]


if renames or root_instance:
    assembly.Assembly.retrieve_assembly = retrieve_assembly
    assembly.Assembly.load_features = load_features

sys.argv = ["onshape-to-robot", *sys.argv[1:]]
main()
