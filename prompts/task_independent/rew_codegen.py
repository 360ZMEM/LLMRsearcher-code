rew_codegen = """
You are an expert in <task_short_desc>, reinforcement learning (RL), and reward function design. We will employ RL methods to complete the task of <desc_short_text>. Here, instructions are provided in **Markdown** format; you should generate a reasonable reward function to meet the task requirements, and the resulting reward function should be formatted as a code block (Python code string in markdown `python ... `).

<Env_desc_wo_obj>

## Objectives

Below are several user requirements that must be satisfied, which may encompass one or more performance metrics. Please note that the requirements are presented in a Python-formatted `dict`, wherein the key represents the name of the user requirement and the value denotes the specific description of that requirement.

```python
<objectives>
```

## Output Guide and Format

You are required to produce specific code for each user requirement, referred to as a "reward component," to implement the reward function, which may include both sparse and dense reward terms. To facilitate user modifications and text matching, you should:

- Follow these steps for each user requirement in order: (a) Analyze how to fulfill this requirement; (b) Analyze how to achieve the goal by identifying the corresponding variables from the `Env` class; (c) Write code snippets according to the preceding analyses, enclosing them in backticks, such as `code write here`.

- Subsequently, generate the reward function based on the preceding analysis. The reward function should be defined within the `Env` class as `def compute_reward(self)`. It should utilize only the variables defined within the `Env` class or create appropriate local variables. Please note that: (a) To avoid the presence of magic numbers in the code, which are difficult to adjust, you should independently output the weights and parameters that may require user modifications; (b) you should separate the reward components of different user requirements using formatted and detailed code comments. For example, when the keys of the objective dictionary from user input are `['foo', 'bar']`, a possible reward function code may be as follows:

  ```python
  import numpy as np
  def compute_reward(self):
      # ------ PARAMETERS ------
      # parameters output independently
      w_xx = 1; # a Weight term for Requirement foo
      r_xxxx = 0.5; # a reward term for Requirement foo
      w_xxx = 2; # a weight term for Requirement bar
      # --------- Code ---------
      # Reward component - doing something (some comment) to fulfill requirement foo
      reward += w_xx * ... # Utilizing parameter variables to calculate reward.
      # Reward component (Another Part Example) - doing something (some comment) to fulfill requirement foo
      reward += r_xxxx * ...
      # Reward component - doing something (some comment) to fulfill requirement bar
      reward += w_xxx * ...
      # (Optional) Reward component - coupled part for requirement foo and bar
      ...
  ```


Then, in order to facilitate text matching, you should further organize the previously generated Python code and create a parameters dictionary and a reward component dictionary, respectively, that strictly comply with the Python `dict` format. Please note that the keys of the two dictionaries must be exactly the same as those in the entered objectives dictionary.

  - **Parameters dictionary:** This dictionary contains the weights and parameters of the reward components corresponding to the requirements.
  
    ```python
    {'foo':['w_xx=1','r_xxxx=0.5'],
     'bar':['w_xxx=2']}
    ```
  
  - **Reward component dictionary:**  This dictionary contains the reward components corresponding to the requirements. Please note that all code strings must retain all indents, line breaks, and other formatting characters of the code presented previously.
  
    ```python
    {'foo':['''    xxx = xxx...
        reward += w_xx * ...''','    reward += r_xxxx * ...'],
     'bar':['    reward += w_xxx * ...']}
    ```

Notice that the final output may contain any number of code blocks; however, the last three must include the complete reward function code, the weight variables dictionary, and the reward components dictionary, in that order.


## Reward Function Requirements and Tips

- If the reward component code for two or more requirements must contain a shared part when designing the reward function, which cannot be decoupled, then the coupling part should be output independently within the code block, and code comments should be used to specify it. When outputting the final dictionaries, codes, and parameters should be filled with the values corresponding to the key `"others"`. However, you should not couple the code for multiple requirements as much as possible, and it is not necessary to set the `"others"` key when there is no coupling.
- Normalize all distance terms by dividing them by the area size, specifically `np.linalg.norm(self.border)`.
- Under the premise of ensuring the necessary degrees of freedom, it is crucial to note that the number of parameters requiring adjustment by the user (i.e., the number of variables defined in the PARAMETERS section) should be kept to a minimum.
- Ensure that your code adheres to the syntax rules of Python and is free of syntax errors. Additionally, only functions from the package `import numpy as np` are permitted for use.
"""
