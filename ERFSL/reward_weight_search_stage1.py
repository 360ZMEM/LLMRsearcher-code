import sys
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import (
    env_desc_prompt,
    task_name,
    desc_short_text,
)
from prompts.task_independent.rew_weight_searcher_stage1 import (
    RWS1_str,
    RWS1_example1,
    RWS1_example2,
)
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec, separate_delimiter
import config  # all config
import re
import time

# ---- dynamic arguments ----
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=1, help="iter-th iterations")
parser.add_argument(
    "--human_feedback", type=str, default="<Empty>", help="Interface for human feedback"
)
args, unknown = parser.parse_known_args()
K = config.K
ITER = args.iter
with open(
    config.REWARD_BASE_DIR + "reward_code_fin.py", "r"
) as f:  # load the reward func file
    reward_func = f.read()
num_desc = num2numspec(K)
RWS1_str = (
    RWS1_str.replace("<num_spec>", num_desc)
    .replace("<Env description>", env_desc_prompt)
    .replace("<rew_function>", reward_func)
    .replace("<human_feedback>,args.human_feedback")
    .replace("<task_name>", task_name)
    .replace("<desc_short_text>", desc_short_text)
)
# generate weight groups and perf summary, delimiter
fin_result_str = ""
with open(config.REPORT_GEN_DIR + f"/train_analysis_ITER{ITER}.md", "r") as f:
    train_log_report = f.read()
# match K summaried output and 1 overall, as we checked in TLA
summarized_output = []
overall_summary = ""
report_lines = train_log_report.splitlines()
for i in range(len(report_lines)):
    if report_lines[i].lower().count("**summarized") > 0:
        summarized_output.append(report_lines[i].rstrip() + "\n")
    if report_lines[i].lower().count("overall summary") > 0:
        overall_summary = report_lines[i:]

for i in range(K):
    perf_str = f"### Group {i+1} " + "\n"
    # load weights
    with open(config.WEIGHT_BASE_DIR + f"weight_ITER{ITER}_REWARD{i + 1}.py", "r") as f:
        weight = f.read()
    perf_str += "```python\n" + weight.rstrip() + "\n```\n"
    # load logs
    with open(
        config.LOG_DIR + f"{config.log_prefix}_ITER{ITER}_REWARD{i + 1}.txt", "r"
    ) as f:
        train_log = f.read()
    _, summarized_logs = separate_delimiter(train_log)
    perf_str += summarized_logs.rstrip() + "\n"
    # match the report
    perf_str += summarized_output[i]
    # finally add the result
    fin_result_str += perf_str
# finally add the overall summary
fin_result_str += overall_summary

RWS1_str = RWS1_str.replace("<weight_training_result>", fin_result_str)
RWS1_str += RWS1_example1 + RWS1_example2
# main
chain = load_LLM_chain(config)
try_time = 1
print(f"Reward Weight Searcher ITER {ITER} (STAGE 1) - Start querying ...")
while True:
    response = chain.invoke({"messages": [HumanMessage(content=RWS1_str)]})
    content = response.content
    # check if there are K lines start with "Suggestion"
    content_lines = content.splitlines()
    count_begg_sugg = 0
    str_begg_sugg = ""
    for i in range(len(content_lines)):
        if content_lines[i].lower().count("**suggestion"):
            count_begg_sugg += 1
            str_begg_sugg += content_lines[i].rstrip() + "\n"
    if count_begg_sugg != K:
        print(
            f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
        )
        try_time += 1
        continue
    # report generation
    filename = config.REPORT_GEN_DIR + f"/rew_weig_search_ITER{ITER}_STAGE1.md"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Query OK! Report has been written into {filename}.")
    if config.verbose == True:
        print(
            f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens"
        )
    break
