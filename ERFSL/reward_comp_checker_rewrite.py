# A variant of Training Log Analyzer (TLA)
import sys
import os
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import env_wo_desc_prompt
from prompts.task_relative.task_objective import objectives,desc_dict, obj_log_dict
from prompts.task_independent.rew_comp_checker import RCC_str
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec, separate_delimiter
import config # all config
import re
import time
import argparse
# dynamic - index of requirement / ITER
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=1, help='iter-th iterations')
parser.add_argument('--req_no', type=str, default='')
args, unknown = parser.parse_known_args()
ITER = args.iter; REQ_NO = list(args.req_no)
RCC_str = RCC_str.replace('<obj_description>',repr(desc_dict))
fin_perf_str = ''
for req_no in REQ_NO:
    with open(config.LOG_DIR, f'{config.log_prefix}_ITER{ITER}_COMP{REQ_NO}.txt') as f:
        perf_str = f.read()
        pattern = r'^.*' + obj_log_dict[objectives[REQ_NO - 1]] + r'.*$'
        matches = re.findall(pattern, perf_str, re.MULTILINE)
    for match in matches:
        fin_perf_str += (match.rstrip() + '\n')
RCC_str = RCC_str.replace('<comp_perf>', fin_perf_str)
chain = load_LLM_chain(config)
try_time = 1
print(f"Reward Component Checker ITER{ITER} - Start querying ...")
while True:
    response = chain.invoke({"messages":[HumanMessage(content=RCC_str)]})
    content = response.content
    # check Python code Block
    pattern_python = r"\```python\n(.+?)\n```"
    matches = re.findall(pattern_python, content, re.DOTALL)
    illegal_match = False
    final_dict = {}
    try:
        final_dict = dict(matches[0])
    except:
        illegal_match = True
    if len(matches) != 1 or illegal_match:
        print(f'The LLM-generated answer can not be normally matched. Retry time {try_time}.')
        try_time += 1
        continue
    # for each requirement, if the answer is '[YES]', then we assert that this component works well
    for idx, val in enumerate(final_dict.values()):
        if val == '[YES]':
            comp_ok_fname = config.REWARD_COMP_DIR + f'comp_ok.txt'
            with open(comp_ok_fname, 'a') as f:
                f.write(str(REQ_NO[idx]) + '\n')
    f_name = config.REPORT_GEN_DIR + f'/reward_comp_check_ITER{ITER}.md'
    with open(f_name,'w') as f:
        f.write(content)
    print(f'Query OK! Report has been written into {f_name}.')
    if config.verbose == True:
        print(f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens")
    break
