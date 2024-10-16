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
parser.add_argument('--req_no', type=int, default=1, help='index of requirement')
args, unknown = parser.parse_known_args()
ITER = args.iter; REQ_NO = args.req_no
partial_obj_desc = '(' + objectives[REQ_NO - 1] + ')' + ' ' + list(desc_dict.values())[REQ_NO - 1]
RCC_str = RCC_str.replace('<partial_obj_description>', partial_obj_desc)
# read the training log
fin_perf_str = ''
with open(config.LOG_DIR, f'{config.log_prefix}_ITER{ITER}_COMP{REQ_NO}.txt') as f:
    perf_str = f.read()
# then get the important part
pattern = r'^.*' + obj_log_dict[objectives[REQ_NO - 1]] + r'.*$'
# match all, there is no need to separate
matches = re.findall(pattern, perf_str, re.MULTILINE)
for match in matches:
    fin_perf_str += (match.rstrip() + '\n')
# then substitute
RCC_str = RCC_str.replace('<comp_perf>', fin_perf_str)
# finally, we can start querying
chain = load_LLM_chain(config)
try_time = 1
print(f"Reward Component Checker ITER{ITER} COMP{REQ_NO} - Start querying ...")
while True:
    response = chain.invoke({"messages":[HumanMessage(content=RCC_str)]})
    content = response.content
    # YES / NO
    pos_count = content.count('[YES]')
    neg_count = content.count('[NO]')
    # generate report
    pattern_summary = r'^.*\*\*' + 'Summarized Output' + r'\*\*.*$'
    matches = re.findall(pattern_summary, content, re.MULTILINE)
    # Supposedly, the length of match should be 1
    if (len(matches) != 1) or (pos_count == neg_count):
        print(f'The LLM-generated answer can not be normally matched. Retry time {try_time}.')
        try_time += 1
        continue
    # if pos_count > neg_count, we assert that this component works well
    if pos_count > neg_count:
        comp_ok_fname = config.REWARD_COMP_DIR + f'comp_ok.txt'
        with open(comp_ok_fname, 'a') as f:
            f.write(str(REQ_NO) + '\n')
    # then we can generate report and output the result.
    f_name = config.REPORT_GEN_DIR + f'/reward_comp_check_ITER{ITER}_COMP_{objectives[REQ_NO-1]}.md'
    with open(f_name,'w') as f:
        f.write(content)
    print(f'Query OK! Report has been written into {f_name}.')
    if config.verbose == True:
        print(f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens")
    break
