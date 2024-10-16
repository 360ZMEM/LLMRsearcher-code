import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
sys.path.append(BASE_DIR)
from prompts.task_relative.env_description import env_desc_prompt
from prompts.desc_meta_prompt import desc_sys_message
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import config

model = ChatOpenAI(model=config.openai_model,openai_api_key=config.openai_api_key,openai_api_base=config.openai_api_base,verbose=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            desc_sys_message,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# then generate response
print("Start querying ...")
response = chain.invoke({"messages":[HumanMessage(content=env_desc_prompt)]})
# generate report
print('------ Answer ------')
print(response.content)
print('------  END  ------')

# write content into report.md
filename = config.REPORT_GEN_DIR + '/env_desc_report.md'
with open(filename,'w') as f:
    f.write(response.content)

print(f'Query OK! Report has been written into {filename}.')
# usage stats
print(f"token_usage: input {response.response_metadata['token_usage']['prompt_tokens']} tokens | output {response.response_metadata['token_usage']['completion_tokens']} tokens | total {response.response_metadata['token_usage']['total_tokens']} tokens")