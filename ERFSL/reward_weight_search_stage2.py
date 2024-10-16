import sys
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import obj_desc_prompt, desc_short_name
from prompts.task_independent.rew_weight_searcher_stage2 import RWS2_str, RWS2_example
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec
import config  # all config
import re
import time

# ---- dynamic arguments ----
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=1, help="iter-th iterations")
args, unknown = parser.parse_known_args()
K = config.K
ITER = args.iter
with open(
    config.REWARD_BASE_DIR + "reward_code_fin.py", "r"
) as f:  # load the reward func file
    reward_func = f.read()

RWS2_str = RWS2_str.replace("<obj_description>", obj_desc_prompt).replace(
    "<task_short_desc>", desc_short_name
)
# add weights & suggestions
weight_str = ""
suggestion_str = ""
for i in range(K):
    w_str = f"Input Group {i+1} " + "\n"
    with open(config.WEIGHT_BASE_DIR + f"weight_ITER{ITER}_REWARD{i + 1}.py", "r") as f:
        weight = f.read()
    w_str += "```python\n" + weight.rstrip() + "\n```\n"
    weight_str += w_str

# similarly, we load the suggestions
sugg_filename = config.REPORT_GEN_DIR + f"/rew_weig_search_ITER{ITER}_STAGE1.md"
with open(sugg_filename, "w") as f:
    sugg_content = f.read()
sugg_lines = sugg_content.splitlines()
for i in range(len(sugg_lines)):
    if sugg_lines[i].lower().startswith("suggestion") or sugg_lines[
        i
    ].lower().startswith("**suggestion"):
        suggestion_str += sugg_lines[i].rstrip() + "\n"
# replace
RWS2_str = RWS2_str.replace("<weight_groups>", weight_str).replace(
    "<suggestions>", suggestion_str
)
RWS2_str += RWS2_example
chain = load_LLM_chain(config)
try_time = 1
print(f"Reward Weight Searcher ITER {ITER} (STAGE 2) - Start querying ...")
while True:
    response = chain.invoke({"messages": [HumanMessage(content=RWS2_str)]})
    content = response.content
    # similar to reward weight initializer, we check the code block
    pattern_python = r"\```python\n(.+?)\n```"
    matches = re.findall(pattern_python, content, re.DOTALL)
    if len(matches) != K:
        print(
            f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
        )
        try_time += 1
        continue
    # we match the code blocks and generate new reward funcs
    for match_idx, match in enumerate(matches):
        try:
            replaced_reward_func = replace_code_block(reward_func, match)
        except:
            print(
                f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
            )
            try_time += 1
            continue
        match_idx += 1
        with open(
            config.REWARD_BASE_DIR + f"reward_ITER{ITER+1}_REWARD{match_idx}.py", "w"
        ) as f:
            f.write(replaced_reward_func)
        with open(
            config.WEIGHT_BASE_DIR + f"weight_ITER{ITER+1}_REWARD{match_idx}.txt", "w"
        ) as f:
            f.write(match)
    f_name = config.REPORT_GEN_DIR + f"/rew_weig_search_ITER{ITER}_STAGE2.md"
    with open(f_name, "w") as f:
        f.write(content)
    print(f"Query OK! Report has been written into {f_name}.")
    # reward functions
    print(f"{K} new reward functions have been written to {config.REWARD_BASE_DIR}.")
    if config.verbose == True:
        print(
            f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens"
        )
    break
