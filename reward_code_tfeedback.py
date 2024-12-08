import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import config
import subprocess
import time
import fnmatch
from plyer import notification
from prompts.task_relative.task_objective import desc_dict, objectives

from utils import generate_reward_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action="store_true")
parser.add_argument("--human_mod", action="store_true")
parser.add_argument("--time_limit", type=int, default=300)
args_global, unknown_args = parser.parse_known_args()
K = config.K
code_delimiter = ["# ------ PARAMETERS ------\n", "# --------- Code ---------\n"]
ITER_GO = 0
LOG_GO = False

while True:
    # check iter_go
    iter_pattern = f"reward_ITER{ITER_GO+1}_COMP*"
    files = os.listdir(config.REWARD_COMP_DIR)
    matching_files = [f for f in files if fnmatch.fnmatch(f, iter_pattern)]
    if len(matching_files) == 0:
        break
    ITER_GO += 1
if (ITER_GO == 0) or (args_global.restart):
    # call the reward_codegen
    args = ["python", f"{config.ERFSL_DIR}/reward_codegen.py"]
    RCG_proc = subprocess.Popen(args)
    RCG_proc.wait()
    ITER_GO = 1
    LOG_GO = False
while True:
    # we read all lines and we get the OK reward comps
    comp_ok_fname = config.REWARD_COMP_DIR + f"comp_ok.txt"
    reward_pass_no = []
    try:
        with open(comp_ok_fname, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                # then convert this line to str
                reward_pass_no.append(int(line))
    except:
        pass
    # end loop
    reward_pass_no = list(set(reward_pass_no))
    if len(reward_pass_no) == len(objectives):
        break
    now_obj_index_list = []  # record the components to be tested
    # Transform the dict code into specific code
    for idx, obj in enumerate(objectives):
        if (idx + 1) in reward_pass_no:
            continue
        now_obj_index_list.append(idx + 1)
        # then check all components and generate python code (we may extract this func to utils.py)
        from Reward_gen.reward_comps.reward_comp_code import reward_comp_code
        from Reward_gen.reward_comps.reward_comp_weight import reward_comp_weight

        reward_func_str = generate_reward_function(
            reward_comp_weight, reward_comp_code, obj, reward_critic=True
        )
        # output reward function
        output_rew_fname = (
            config.REWARD_COMP_DIR + f"reward_ITER{ITER_GO}_COMP{idx+1}.py"
        )
        with open(output_rew_fname, "w") as f:
            f.write(reward_func_str)
    # if there are more than K components, we test the components separately
    for i in range((len(now_obj_index_list) - 1) // K + 1):
        # run_train.py
        args = (
            [
                "python",
                f"{config.ERFSL_DIR}/run_train.py",
                "--train_components",
                "--train_comp_idx",
                str(now_obj_index_list[i * K: (i + 1) * K]),
            ]
            + config.arguments
            + unknown_args
        )
        training_proc = subprocess.Popen(args)
        training_proc.wait()
    args = [
        "python",
        f"{config.ERFSL_DIR}/reward_comp_checker.py",
        "--iter",
        str(ITER_GO),
        "req_no",
        repr(now_obj_index_list),
    ]
    rcc_proc = subprocess.Popen(args)
    rcc_proc.wait()
    args = ["python", f"{config.ERFSL_DIR}/reward_critic.py", "--iter", str(ITER_GO)]
    RC_proc = subprocess.Popen(args)
    RC_proc.wait()
    # add ITER_GO
    ITER_GO += 1

from Reward_gen.reward_comps.reward_comp_code import reward_comp_code
from Reward_gen.reward_comps.reward_comp_weight import reward_comp_weight

# finally, we write the reward code to `reward_code_fin.py` (out the while loop)
output_rew_fname = config.REWARD_BASE_DIR + "reward_code_fin.py"
with open(output_rew_fname, "w") as f:
    f.write(reward_func_str)
