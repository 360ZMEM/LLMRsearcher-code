TLA_str = """
We provide training logs for several reward functions, taken from a data-collection reinforcement learning (RL) task. To guide the user in further tuning the parameters of the reward function, you need to write a summary for each log showing the key message, based on user requirements detailed below.

## User requirements

The main objectives to be optimized for the task are: 

<obj_description>

## Output guide

There are multiple sets of weight parameters and the performance training logs that are generated by training policies using these weight parameters. You should first consider summarizing each group of performance results individually, and then compare them with each other.

## Performance result

Here is the performance log you need to analyze, [HGH] means higher is better, while [LOW] means lower is better.
"""  # insert the result
