rew_critic = """
You are an expert in <task_short_desc>, reinforcement learning (RL), and the design of reward components. I am designing a reward function for the task of <desc_short_text>, and the reward components of this function, which fulfill user requirements, have been tested. However, there are some components that do not work properly and cannot fulfill the corresponding user requirements. You are required to inspect the reward components to identify the issues. Subsequently, analyze how to effectively modify the reward components to align them with the task requirements and to guide the training strategy effectively.

<Env_desc_wo_obj>

## Objectives of the Task

The following Python dictionary specifies the descriptions of user requirements, where the keys represent the names of the requirements and the values provide their descriptions.

```python
<objectives>
```

## Reward Component Codes and Outcomes

The following three Python code blocks contain the reward component code for all user requirements. Please note that all reward components are tested separately; specifically, when evaluating a reward component, the weights of all remaining requirements are set to zero during training.

**reward components:** This dictionary contains the Python code for the reward components corresponding to each requirement.

```python
<rew_comp_code>
```

**Paremeters:** This dictionary contains the parameters and variables of the reward component code that were present previously, whose values may require human modifications.

```python
<rew_comp_weight>
```

**Training results:** This dictionary describes whether each reward component meets its corresponding user requirement.

```python
<run_result>
```

## Output the Result Step-by-step

You should identify why each reward component does not work properly, one by one, and rectify them. Specifically,

- Analyze why the reward components cannot meet user demands. This may be due to an unreasonable format of the reward component, or it may arise from errors within the reward component. You should identify these issues.
- Carefully reanalyze the environment code to identify which variables, as part of the reward component, can guide the strategy for task completion. Focus on breaking down the task requirements. A step-by-step analysis and listing of potential contributing factors, along with code snippets (enclosed in backticks, such as code write here), may assist in accurately analyzing the reasons.
- Summarize these principles and present a revised and appropriate reward component that addresses this user requirement. 

For other reward components that function properly, no output analysis is necessary, and the corresponding reward components and parameters should remain unchanged. However, if changes must also be made to the reward components of other requirements due to coupling, you should explicitly specify this and provide the code blocks of the reward components that need to be rectified and coupled, respectively.

## Output Format Constraint

The output should analyze and rectify each reward component that cannot meet the corresponding user requirements, one by one. When rectifying a reward component, ensure that the indents and line breaks of the original code are preserved; output new Python code blocks. To avoid magic numbers in the code, you must output the weights and parameters in a separate code block. Refer to the section titled "Format Example" for specific formatting guidelines.

After you have output all code blocks of the reward components, you should reformat them into two dictionaries: one for Python code and one for parameters, respectively, similar to the code blocks provided in the section titled "Reward Component Codes and Outcomes." Specifically, first populate each reward component code string with the corresponding values of the requirements to complete the formatted dictionary of reward components, ensuring that all indentation (spaces) and line breaks (`\n`) are retained. Subsequently, populate the parameters with the corresponding values of the requirements to complete the formatted dictionary of parameters.

## Note

- All reward components are derived from a reward function in the `Env` class, formatted as `def compute_reward(self)`. The reward function initializes `reward = np.zeros(self.N_AUV)` and ultimately executes `return reward` as the reward value.
-  Normalize all distance terms by dividing them by the area size (specifically, `np.linalg.norm(self.border)`).
- In your modified version, ensure that the numerical values of the generated code are of the same order of magnitude. Specifically, it is not permissible to modify the original code solely by the relevant coefficients.
- The number of parameters that must be adjusted by the user (i.e., the number of variables defined in the `PARAMETERS` section) should be minimized.
- If you believe that the variables or functions in the task description code are insufficient to rewrite the reward component, you may create these variables; however, you should instruct the user to complete them.
"""

rew_critic_example = """
## Format Example

To avoid revealing suggestions, user requirements will be replaced with requirement foo, requirement bar, etc.

### Example user input

**Reward components: **

```python
{'foo':['''    xxx = xxx...
    reward += w_xx * ... # XXXX''',...],'bar':['    reward += r_xxxx * ...'],...}
```

**Parameters: **

```python
{'foo':['w_xx=1','r_xxx=0.5'],'bar':['r_xxxx=2'],...}
```

**Training results: **

```python
{'foo':'[YES]','bar':'[NO]',...}
```

### Example output

**User requirements that have not been satisfied: ** `['bar',...]`

**Analysis of improperly designed reward components in order:**

**Requirement 'bar'**:

- Main reasons for failure:
  - Sparse reward: The current reward component only provides feedback when the variable xxx ... This implies that the agent only receives feedback when ...
  - Failure to consider XXX: This oversight can result in the agent neglecting risk factors such as XXX...
  - ... (other issues)
- Identify variables to guide strategy: 
  - A suggestion for addressing the first issue is that the reward function should be combined with distance factors to create dense feedback, which may relate to the agent's position `xxx` and XXX in the environment, as well as...
  
    `code_snippet_example`
  
  - A suggestion for the second issue is that we should take into account xxx...
  
  - ... (Other suggestions for issues)
- Revised reward component (keep indents)

```python
	# ------ PARAMETERS ------
    w_another = 0.8
    r_xxxx = 2
    # --------- Code ---------
    reward += w_another * ... # some comment
    reward += r_xxxx * ...
```

### Reformatted dictionaries

**Reward components:** The components and parameters of requirement foo remain consistent with the original input. In the generated dictionaries, the value corresponding to each requirement is represented as a list, which is utilized to contain multiple parameters or components. For the reward component, the Python statement string is surrounded by triple quotes, and all indents and line breaks of the generated code are preserved.

```python
{'foo':['''    xxx = xxx...
    reward += w_xx * ... # XXXX''',...],'bar':['''    reward += w_another * ...
    reward += r_xxxx * ...'''],...}
```

**Parameters:** 

```python
{'foo':['w_xx=1','r_xxx=0.5'],'bar':['w_another=0.8','r_xxxx=2'],...}
```
"""
