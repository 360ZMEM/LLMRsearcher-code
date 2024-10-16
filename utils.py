from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
import inflect
import torch
import multiprocessing
import sys
import time

def num2numspec(num): # e.g. input 5 output str '**5(five)**'
    inf_engine = inflect.engine()
    K_word = inf_engine.number_to_words(num)
    return '**' + str(num) + '(' + K_word + ')' + '**'

def replace_code_block(original_text, new_text):
    # Define the patterns to find the part to replace
    part1_pattern = r"(# ------ PARAMETERS ------\n)([\s\S]*?)(# --------- Code ---------)"
    
    # Find the original indentation level by searching for the first line after PART1
    match = re.search(part1_pattern, original_text)
    if not match:
        print("Code Match Error!")
        raise ValueError
    
    part1_start, original_content, part2_start = match.groups()
    
    # Determine the indentation level of the original content
    lines = original_content.splitlines()
    if lines:
        first_line = lines[0]
        original_indent = len(first_line) - len(first_line.lstrip(' '))
    else:
        original_indent = 0
    
    # Prepare the new text with the correct indentation
    new_lines = new_text.splitlines()
    indented_new_text = '\n'.join(' ' * original_indent + line.strip() for line in new_lines)
    
    # Replace the original content with the new indented content
    replaced_text = original_text.replace(original_content, indented_new_text + '\n    ')
    
    return replaced_text

def generate_reward_function(weight, code, objectives, reward_critic = True):
    reward_func_str = 'def compute_reward(self):\n    # ------ PARAMETERS ------\n'
    weight_fin_str = ''; code_fin_str = ''
    objectives = [objectives] if (type(objectives) != list) else objectives
    for obj in objectives:
        for idx, weight_str in enumerate(weight[obj]):
            weight_str = (weight_str.split('=') + ';') if reward_critic == False else weight_str
            comment_str = '# ' + 'weight ' if weight_str == ['w'] else 'reward(penalty) ' + f'term for {obj} (term {idx+1})' if reward_critic == False else ''
            weight_str = ' ' * 4 + weight_str + comment_str + '\n'
            weight_fin_str += weight_str
        for idx, code_str in enumerate(code[obj]):
            code_fin_str += code_str.rstrip() + '\n'
    reward_func_str += (weight_fin_str) + '    # --------- Code ---------\n' + code_fin_str
    return reward_func_str

# dynamically waiting for user input

def user_input():
    user_input_value = input()
    if user_input_value == '':
        return 'No'
    return user_input_value

def dynamic_waiting_input(timeout):
    manager = multiprocessing.Manager()
    result = manager.Value(str, "")
    proc = multiprocessing.Process(target=lambda r: r.set(user_input()), args=(result,))
    proc.start()
    start_time = time.time()
    while True:
        print(f'\r User respond remaining time {timeout}s.')
        time.sleep(0.05)
        if (time.time() - start_time) > 1:
            start_time = time.time()
            timeout -= 1
        if result.value != '':
            value = result.value 
            proc.terminate()
            return value
        if timeout < 0:
            proc.terminate()
            return ''
        

def separate_delimiter(original_text, delimiter='\n--------\n'):
    pattern = r'^(.*?)' + delimiter + r'(.*)$'
    match = re.match(pattern, original_text, re.DOTALL)
    if not match:
        print("Log Match Error!")
        raise ValueError
    part1 = match.group(1).strip()
    part2 = match.group(2).strip()
    return part1, part2

def load_LLM_chain(config):
    if not config.opensource_model:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=config.openai_model,temperature=config.temperature,top_p=config.Top_P,openai_api_key=config.openai_api_key,openai_api_base=config.openai_api_base,verbose=True)
    # huggingface LLM
    else:
        from langchain_huggingface.llms import HuggingFacePipeline
        from transformers import pipeline
        hf_pipe = pipeline("text-generation", model=config.opensource_model, torch_dtype=torch.bfloat16, device_map="auto",temperature=config.temperature,top_p=config.Top_P)
        # generate model
        model = HuggingFacePipeline(pipeline=hf_pipe)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You can help me by answering my questions.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model
    return chain