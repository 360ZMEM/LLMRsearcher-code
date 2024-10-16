import sys
import os
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import env_wo_desc_prompt
from prompts.task_relative.task_objective import desc_dict, objectives
from prompts.task_independent.rew_codegen import rew_codegen
from langchain_core.messages import HumanMessage
from utils import replace_code_block, load_LLM_chain, num2numspec
import config # all config
import re
import time
# substitude, only env desc
rew_codegen = rew_codegen.replace('<Env_desc_wo_obj>',env_wo_desc_prompt).replace('<objectives>',repr(desc_dict))
chain = load_LLM_chain(config)
# in main loop, we should extract all components
print(f"Reward Code Generator - Start querying ...")
try_time = 1
while True:
    response = chain.invoke({"messages":[HumanMessage(content=rew_codegen)]})
    content = response.content
    pattern_python = r"\```python\n(.+?)\n```"
    matches = re.findall(pattern_python, content, re.DOTALL)
    illegal_match = False
    for idx, match in enumerate(matches):
        try:
            in_dict = dict(match)
            if sorted(list(in_dict.keys())) != sorted(objectives):
                raise ValueError
        except:
            illegal_match = True if idx != 0 else False
    if len(matches) != 2 or illegal_match:
        print(f'The LLM-generated answer can not be normally matched. Retry time {try_time}.')
        try_time += 1
        continue
    # match 0 -> weights, match 1 -> reward codes
    with open(config.REWARD_COMP_DIR + '/reward_comp_weight.py', 'w') as f:
        f.write('reward_comp_weight = ' + matches[1].rstrip() + '\n')
    with open(config.REWARD_COMP_DIR + '/reward_comp_code.py', 'w') as f:
        f.write('reward_comp_code = ' + matches[2].rstrip() + '\n')
    # generate report
    f_name = config.REPORT_GEN_DIR + f'reward_code_gen_{int(time.time())}.md'
    with open(config.REPORT_GEN_DIR+f_name, 'w') as f:
        f.write(content)
    print(f'Initial Weight Groups generate OK! Report generated in {f_name} .')
    if config.verbose == True:
        print(f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens")
    break


