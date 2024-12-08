import os

# Specify your OpenAI keys and api base address (if any) here
openai_api_key = "your_api_key"
openai_api_base = "https://api.openai.com/v1"
# Also, you can specify the model
openai_model = "gpt-4o-2024-08-06"  # gp4-4o-mini
opensource_model = None  # 'meta-llama/Meta-Llama-3-70B' / 'Qwen/Qwen2.5-72B' / ...
log_dir = f"/logs"  # relative dir
log_prefix = "TD3"
temperature = 0.5
Top_P = 1
K = 5
verbose = True
# dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERFSL_DIR = BASE_DIR + "/"
REWARD_BASE_DIR = BASE_DIR + "/Reward_gen/reward_funcs/"
WEIGHT_BASE_DIR = BASE_DIR + "/Reward_gen/weights/"
REPORT_GEN_DIR = BASE_DIR + "/Reward_gen/reports/"
REWARD_COMP_DIR = BASE_DIR + "/Reward_gen/reward_comps/"
LOG_DIR = BASE_DIR + "/Reward_gen/logs/"
DIRS = [REWARD_BASE_DIR, WEIGHT_BASE_DIR, REPORT_GEN_DIR, LOG_DIR, REWARD_COMP_DIR]
for DIR in DIRS:
    if not os.path.exists(DIR):
        os.makedirs(DIR)
# additional arguments for train_td3.py
arguments = []  # e.g. ['--batch_size','64','--hidden_size','128', ...]