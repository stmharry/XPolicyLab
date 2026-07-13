#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -c '
import sys, os, json

import yaml

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]
xpl_utils_dir = sys.argv[3]

official_env = os.path.join(os.path.dirname(xpl_utils_dir), "env_cfg", f"{env_cfg_type}.yml")
official_robots = os.path.join(os.path.dirname(xpl_utils_dir), "env_cfg", "robot", "_robot_info.json")
robodojo_env = os.path.join(root_dir, "configs", "environment", f"{env_cfg_type}.yml")
robodojo_robots = os.path.join(root_dir, "configs", "robot", "_robot_info.json")
local_profiles = os.path.join(xpl_utils_dir, "robot", "_robot_info.json")

if os.path.isfile(official_env) and os.path.isfile(official_robots):
    env = yaml.safe_load(open(official_env, "r", encoding="utf-8"))
    robot_action_dim_info = json.load(open(official_robots, "r", encoding="utf-8"))[env["config"]["robot"]]
elif os.path.isfile(robodojo_env) and os.path.isfile(robodojo_robots):
    env = yaml.safe_load(open(robodojo_env, "r", encoding="utf-8"))
    robot_action_dim_info = json.load(open(robodojo_robots, "r", encoding="utf-8"))[env["config"]["robot"]]
else:
    robot_action_dim_info = json.load(open(local_profiles, "r", encoding="utf-8"))[env_cfg_type]

print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}" "${SCRIPT_DIR}"
