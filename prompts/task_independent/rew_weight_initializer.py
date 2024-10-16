RWI_str = """
You are an expert in <task_short_desc>, reinforcement learning (RL), and reward function design. We are designing reward functions to utilize RL methods to complete the task of multiple AUVs in collecting information from underwater sensor nodes (SNs). This task involves multiple optimization objectives. However, it is important to ensure a balanced weighting of the reward components corresponding to these objectives. Now, you need to calculate the values of the reward components and generate possible values for <num_spec> of parameters based on these calculations.

<Env_description>

## Task Guide

The code for the reward function is as follows:

```python
<rew_function>
```

Now, complete the values of the corresponding weights in the form of a Python code block, following the format below:

```python
# Set X: focusing on some requirements / balanced
w_collision = xxx; # some comments for choosing this value
w_border = xxx;
w_service = xxx;
w_overflow = xxx;
w_energy = xxx;
```

Here are guides for choosing the value for parameters:

- Ensure that the weighted sum of all reward components is approximately equal in magnitude and that they exhibit similar changes with respect to actions. To achieve this, you should output a preliminary calculation of the value of each reward component based on the given or custom example value. For reward components with conditional judgments (such as collision penalties), you can set a value that satisfies the penalty condition (for example: `dist = 5m`). Note the regularization term `/ np.linalg.norm(self.border)`. Finally, the weighted reward components should have the same scale.
- The returned reward values should be on the order of magnitude of $10^1$.

## Requirements

- You can only assign values to the given parameters and cannot add or remove any.
- The first set of parameters should be the one most likely to satisfy all user requirements. It should balance the weighted values of the reward components as much as possible. The remaining sets can involve scaling certain parameter values, with a minimum step size of 5x. Note that reducing the weight of some parameters means that the relative change in the remaining weights will increase, or vice versa. You should pay attention to the ratio changes between parameters and try to minimize the number of parameters adjusted.
- The weights for some components may be negative, thereby penalizing or rewarding certain behaviors, so ensure you do not get the signs wrong.
- The final output must be <num_spec> Python code blocks similar to the above example. No more Python code blocks are allowed.
"""
