# multi-process parallel training (default K=5)
import argparse
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(BASE_DIR)
import config
import subprocess
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=1, help="iter-th iterations")
parser.add_argument("--episode_num", type=int, default=480)
parser.add_argument(
    "--train_components",
    action="store_true",
    help="The reward function is a single component, rather than a integral one.",
)
parser.add_argument(
    "--train_comp_idx", type=str, default="", help="Indexes of components."
)
(
    args,
    unknown_args,
) = parser.parse_known_args()  # you can specify other arguments of train_td3.py
K = config.K
ep_num = args.episode_num
# if train_components is specified, argument K will be overridden
if args.train_components:
    train_comp_idx = list(args.train_comp_idx)
    K = len(train_comp_idx)
# run subprocesses one by one
proc = []
ep = np.ones(args.K)
proc_alive = np.zeros(args.K)
for i in range(K):
    comp_args = ["--train_components", args.train_components, "--reward_no"]
    comp_args += [str(train_comp_idx[i])] if args.train_components else [str(i + 1)]
    args = (
        [
            "python",
            f"{BASE_DIR}/RL_task/train_td3.py",
            "--iter",
            str(args.iter),
            "--episode_num",
            str(ep_num),
        ]
        + comp_args
        + unknown_args
    )
    proc.append(
        subprocess.Popen(args, stdout=subprocess.PIPE)
    )  # hide the output of train_td3.py
# START TRAINING
print(f"Start Training! Reward weight - ITER{args.iter} K={K}")
# Then monitoring the training status
while True:
    time.sleep(0.05)
    print_str = ""
    # monitoring the subprocess output
    for i in range(K):
        while True:
            line = proc[i].stdout.readline()
            if not line:
                break
            ep[i] += 1
        # is alive
        poll = proc[i].poll()
        proc_alive[i] = poll is None  # True if it's not alive
        print_str += f"REWARD {i} - EP {ep[i]} / {ep_num} "
        if i != K - 1:
            print_str += "|"
    # print the training progress
    print(print_str, end="\r")
    # if all process is not alive
    if np.sum(proc_alive) == 0:
        break
