import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import config
import subprocess
import time
import argparse
from plyer import notification
from utils import dynamic_waiting_input

parser = argparse.ArgumentParser()
parser.add_argument('--restart',action='store_true')
parser.add_argument('--human_mod',action='store_true')
parser.add_argument('--quiet',action='store_true', help='Specify this to not present the performance summary.')
parser.add_argument('--time_limit', type=int, default=300)
args, unknown_args = parser.parse_known_args()
K = config.K

# first check the process
ITER_GO = 0 # this check the weight generation status
LOG_GO = False # this check the log reading status (override by `--restart`)
while True:
    iter_fname = config.WEIGHT_BASE_DIR + f'weight_ITER{ITER_GO+1}_REWARD1.py'
    if not os.path.exists(iter_fname):
        break
    ITER_GO += 1
    
# Also, we check if the corresponding log file exists
if (ITER_GO == 0) or (args.restart):
    # call the reward weight initializer
    args = ['python',f'{config.ERFSL_DIR}/rew_weight_initializer.py']
    RWI_proc = subprocess.Popen(args) # we don't specify STDOUT
    RWI_proc.wait()
    ITER_GO = 1; LOG_GO = False
else:
    log_fname = config.LOG_DIR + f'{config.log_prefix}_ITER{ITER_GO}_REWARD1.txt'
    if os.path.exists(log_fname):
        LOG_GO = True
        print('Log file detected. Directly call the reward weight searcher.')

# then execute the main loop
while True:
    # if there is no log, we run train.py
    if not LOG_GO:
        args = ['python',f'{config.ERFSL_DIR}/run_train.py'] + config.arguments + unknown_args
        training_proc = subprocess.Popen(args)
        training_proc.wait()
        print(f'------ Training ITER {ITER_GO} OK! ------')
    else:
        LOG_GO = False
    # we call the training log analyzer, and show the performance report
    args = ['python', f'{config.ERFSL_DIR}/training_log_analyzer.py', 
            '--iter', str(ITER_GO)]
    TLA_proc = subprocess.Popen(args)
    TLA_proc.wait()
    # notification
    notification.notify(
        title = f'ERFSL - Training ITER{ITER_GO} OK',
        message = 'You may input your feedback',
        app_icon = config.ERFSL_DIR + 'icon.png',
        timeout = 10,
    )
    # then, we can set a entry point for user to type in the feedback
    print(f'Training ITER{ITER_GO} OK. Do you want to input feedback? If yes, input Y and enter. If no, directly press enter or input any other words.\n')
    ret_str = dynamic_waiting_input()
    if ret_str.lower().strip() not in ['y','yes']:
        feedback_str = '<Empty>'
    else:
        feedback_str = input('Please enter Your feedback.')
    args = ['python', f'{config.ERFSL_DIR}/reward_weight_search_stage1.py', 
            '--iter', str(ITER_GO), '--human_feedback',feedback_str]
    RWS_stage1_proc = subprocess.Popen(args)
    RWS_stage1_proc.wait()
    print(f'------ Reward weight Searcher STAGE1 ITER {ITER_GO} OK! ------')
    # then we call the subprocess 2
    args = ['python', f'{config.ERFSL_DIR}/reward_weight_search_stage2.py', 
            '--iter', str(ITER_GO)]
    RWS_stage2_proc = subprocess.Popen(args)
    RWS_stage2_proc.wait()
    print(f'------ Reward weight Searcher STAGE2 ITER {ITER_GO} OK! ------')
    print(f'Start next iteration ITER {ITER_GO} -> {ITER_GO+1}')
    ITER_GO += 1

    
