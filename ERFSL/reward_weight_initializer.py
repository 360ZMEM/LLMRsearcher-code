import sys
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import (
    env_desc_prompt,
    task_name,
    desc_short_text,
)
from prompts.task_independent.rew_weight_initializer import RWI_str
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec
import config  # all config
import re
import time

K = config.K
# load reward func as str
with open(
    config.REWARD_BASE_DIR + "reward_code_fin.py", "r"
) as f:  # load the reward func file
    reward_func = f.read()
# generate a number expression e.g.  5 => **5(five)**
num_desc = num2numspec(K)
# substitute env and reward placeholders to specific code/desc
RWI_str = (
    RWI_str.replace("<num_spec>", num_desc)
    .replace("<Env_description>", env_desc_prompt)
    .replace("<rew_function>", reward_func)
    .replace("<task_name>", task_name)
    .replace("<desc_short_text>", desc_short_text)
)
# then query OpenAI API

chain = load_LLM_chain(config)

try_time = 1
print("Reward Weight Initializer - Start querying ...")
while True:
    response = chain.invoke({"messages": [HumanMessage(content=RWI_str)]})
    content = response.content
    # match all Python code block
    pattern_python = r"\```python\n(.+?)\n```"  # start with ```python and end with ```
    matches = re.findall(pattern_python, content, re.DOTALL)
    # check the length of matches. if not args.K, regenerate
    if len(matches) != K:
        print(
            f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
        )
        try_time += 1
        continue
    # then generate new reward funcs for first iteration and replace
    for match_idx, match in enumerate(matches):
        try:
            replaced_reward_func = replace_code_block(reward_func, match)
        except:
            print(
                f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
            )
            try_time += 1
            continue
        # generate new funcs
        match_idx += 1
        with open(
            config.REWARD_BASE_DIR + f"reward_ITER1_REWARD{match_idx}.py", "w"
        ) as f:
            f.write(replaced_reward_func)
        with open(
            config.WEIGHT_BASE_DIR + f"weight_ITER1_REWARD{match_idx}.txt", "w"
        ) as f:
            f.write(match)
        # if the output is OK, generate the report
    f_name = config.REPORT_GEN_DIR + f"reward_weight_initializer_{int(time.time())}.md"
    # output report dir
    with open(f_name, "w") as f:
        f.write(content)
    print(f"Initial Weight Groups generate OK! Report generated in {f_name} .")
    print(f"{K} new reward functions have been written to {config.REWARD_BASE_DIR}.")
    if config.verbose == True:
        print(
            f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens"
        )
    break
