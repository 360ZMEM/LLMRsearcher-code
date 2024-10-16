import sys
import os
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import obj_desc_prompt
from prompts.task_independent.training_log_analyzer import TLA_str
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec, separate_delimiter
import config # all config
import re
import time
# ---- dynamic arguments ----
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=1, help='iter-th iterations')
parser.add_argument('--show_performance', action='store_true')
args, unknown = parser.parse_known_args()
K = config.K;  ITER = args.iter
TLA_str = TLA_str.replace('<obj_description>',obj_desc_prompt)
# add performance results
for i in range(K):
    perf_str = f'### Group {i+1} ' + '\n'
    # load weight py
    with open(config.WEIGHT_BASE_DIR + f'weight_ITER{ITER}_REWARD{i + 1}.py','r') as f:
        weight = f.read()
    perf_str += "```python\n" + weight.rstrip() + '\n```\n'
    # load perf 
    with open(config.LOG_DIR + f'{config.log_prefix}_ITER{ITER}_REWARD{i + 1}.txt','r') as f:
        train_log = f.read()
    train_log, _ = separate_delimiter(train_log)
    perf_str += train_log.rstrip() + '\n'
TLA_str.replace("<weights_and_pref_res>",perf_str)
chain = load_LLM_chain(config)
try_time = 1
print(f"Training Log Analyzer ITER {ITER} - Start querying ...")
while True:
    response = chain.invoke({"messages":[HumanMessage(content=TLA_str)]})
    content = response.content
    # check if there are 5 summarizations and one overall summary (case insensitive)
    if (content.lower().count('**summarized') != 5) or (content.lower().count('overall summary') != 1):
        print(f'The LLM-generated answer can not be normally matched. Retry time {try_time}.')
        try_time += 1
        continue
    # only generate report if keywords can be found
    filename = config.REPORT_GEN_DIR + f'/train_analysis_ITER{ITER}.md'
    # save the file to the filename
    with open(filename,'w') as f:
        f.write(content)
    print(f'Query OK! Report has been written into {filename}.')
    if config.verbose == True:
        print(f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens")
    # show performance 
    if args.show_performance:
        print(f'------ TRAINING RESULT ITER {ITER} ------')
        print(perf_str)
        print(f'------  TRAINING RESULT SHOW OK   ------')
    break
