import sys
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import env_wo_desc_prompt
from prompts.task_relative.task_objective import objectives, desc_dict
from prompts.task_relative.env_description import desc_short_text, task_name
from prompts.task_independent.rew_critic import rew_critic, rew_critic_example
from Reward_gen.reward_comps.reward_comp_code import reward_comp_code
from Reward_gen.reward_comps.reward_comp_weight import reward_comp_weight
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec
import config  # all config
import re
import time
import argparse
import ast

# ITER / Req. No
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=1, help="iter-th iterations")
args, unknown = parser.parse_known_args()
ITER = args.iter
rew_critic = (
    rew_critic.replace("<Env_desc_wo_obj>", env_wo_desc_prompt)
    .replace("<objectives>", repr(desc_dict))
    .replace("<rew_comp_weight>", repr(reward_comp_weight))
    .replace("<rew_comp_code>", repr(reward_comp_code))
    .replace("<task_short_desc>", desc_short_text)
    .replace("<task_name>", task_name)
)
rcc_report_name = config.REPORT_GEN_DIR + f"/reward_comp_check_ITER{ITER}.md"
with open(rcc_report_name, "r") as f:
    rcc_report_str = f.read()
pattern_python = r"\```python\n(.+?)\n```"
# match code block
matches = re.findall(pattern_python, rcc_report_str, re.DOTALL)
rew_critic = rew_critic.replace("<run_result>", matches[0])
rew_critic += rew_critic_example
chain = load_LLM_chain(config)
# in main loop, we should extract all components
print(f"Reward Critic ITER{ITER} - Start querying ...")
try_time = 1
while True:
    response = chain.invoke({"messages": [HumanMessage(content=rew_critic)]})
    content = response.content
    matches = re.findall(pattern_python, content, re.DOTALL)
    illegal_match = False
    for idx, match in enumerate(matches[-2:]):
        try:
            in_dict = ast.literal_eval(match)
            if (type(in_dict) != dict) or (not set(in_dict.keys()) >= set(objectives)):
                raise ValueError
        except:
            illegal_match = True if idx != 0 else False
    if illegal_match:
        print(
            f"The LLM-generated answer can not be normally matched. Retry time {try_time}."
        )
        try_time += 1
        continue
    with open(config.REWARD_COMP_DIR + "/reward_comp_weight.py", "w") as f:
        f.write("reward_comp_weight = " + matches[1].rstrip() + "\n")
    with open(config.REWARD_COMP_DIR + "/reward_comp_code.py", "w") as f:
        f.write("reward_comp_code = " + matches[2].rstrip() + "\n")
    # generate report
    f_name = config.REPORT_GEN_DIR + f"/reward_critic_ITER{ITER}.md"
    with open(f_name, "w") as f:
        f.write(content)
    print(f"Query OK! Report has been written into {f_name}.")
    if config.verbose == True:
        print(
            f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens"
        )
    break
