"""
Build the full ECHO URDF: take the body export (arms are rigid blocks there)
and replace each rigid arm with the jointed arm export.

Both exports must have been run with --save-pickle:
    .venv-onshape/bin/python tools/onshape_export.py export/arm --save-pickle
    .venv-onshape/bin/python tools/onshape_export.py export/body --save-pickle
    .venv-onshape/bin/python tools/merge_arms.py

Writes export/echo/robot.urdf with meshes in export/echo/assets.
"""

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from onshape_to_robot.config import Config
from onshape_to_robot.exporter_urdf import ExporterURDF

REPO = Path(__file__).resolve().parent.parent
BODY_DIR = REPO / "export" / "body"
ARM_DIR = REPO / "export" / "arm"
OUT_DIR = REPO / "export" / "echo"

# Name prefix for the arm copy -> shoulder joint (in the body export) whose
# child link is that rigid arm
ARMS = {"right": "right_shoulder", "left": "left_shoulder"}
# Arm link that the next body (the hand) attaches to
ARM_TIP_LINK = "rs00"
# Matched parts must agree on the arm placement within this tolerance (m)
PLACEMENT_TOLERANCE = 1e-4


def mesh_hash(export_dir: Path, part) -> str | None:
    if not part.meshes:
        return None
    return hashlib.sha1((export_dir / part.meshes[0].filename).read_bytes()).hexdigest()


def arm_placement(rigid_link, arm_robot) -> np.ndarray:
    """
    Transform from the arm export's frame to the robot's world frame, found by
    matching identical meshes between the rigid arm and the arm export.
    """
    arm_parts = {}
    for link in arm_robot.links:
        for part in link.parts:
            arm_parts.setdefault(mesh_hash(ARM_DIR, part), []).append(part)

    candidates = []
    for part in rigid_link.parts:
        matches = arm_parts.get(mesh_hash(BODY_DIR, part), [])
        if len(matches) == 1:
            candidates.append(part.T_world_part @ np.linalg.inv(matches[0].T_world_part))

    if not candidates:
        sys.exit(f"No part of {rigid_link.name} matches the arm export")
    reference = candidates[0]
    spread = max(np.abs(candidate - reference).max() for candidate in candidates)
    print(f"  placed from {len(candidates)} matching parts, max disagreement {spread:.2e}")
    if spread > PLACEMENT_TOLERANCE:
        sys.exit("Matched parts disagree on the arm placement; the arm versions differ")
    return reference


def load(export_dir: Path):
    with open(export_dir / "robot.pkl", "rb") as stream:
        return pickle.load(stream)


robot = load(BODY_DIR)

for prefix, shoulder_joint_name in ARMS.items():
    print(f"* Attaching {prefix} arm")
    arm = load(ARM_DIR)
    shoulder_joint = robot.get_joint(shoulder_joint_name)
    rigid_link = shoulder_joint.child
    T_world_arm = arm_placement(rigid_link, arm)

    for link in arm.links:
        link.name = f"{prefix}_{link.name}"
        for part in link.parts:
            part.T_world_part = T_world_arm @ part.T_world_part
            for mesh in part.meshes:
                mesh.filename = mesh.filename.replace("assets/", "assets/arm/", 1)
        link.frames = {
            f"{prefix}_{name}": T_world_arm @ T for name, T in link.frames.items()
        }
    for joint in arm.joints:
        joint.name = f"{prefix}_{joint.name}"
        joint.T_world_joint = T_world_arm @ joint.T_world_joint

    tip_link = arm.get_link(f"{prefix}_{ARM_TIP_LINK}")
    for joint in robot.get_link_joints(rigid_link):
        joint.parent = tip_link
    shoulder_joint.child = arm.base_links[0]

    robot.links.remove(rigid_link)
    robot.links.extend(arm.links)
    robot.joints.extend(arm.joints)

# Output directory with both mesh sets
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
shutil.copytree(BODY_DIR / "assets", OUT_DIR / "assets")
shutil.copytree(ARM_DIR / "assets", OUT_DIR / "assets" / "arm")
body_config = json.loads((BODY_DIR / "config.json").read_text())
(OUT_DIR / "config.json").write_text(json.dumps(body_config, indent=4))

config = Config(str(OUT_DIR))
for processor in config.processors:
    processor.process(robot)
ExporterURDF(config).write_xml(robot, str(OUT_DIR / "robot.urdf"))
print(f"* Wrote {OUT_DIR / 'robot.urdf'}: {len(robot.links)} links, {len(robot.joints)} joints")
