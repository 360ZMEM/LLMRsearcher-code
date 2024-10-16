RWS1_str = '''
You are an expert in <task_short_desc>, reinforcement learning (RL), and reward function design. We have designed reward functions to complete <task_short_desc>. However, since these reward functions need to satisfy multiple user requirements, the weight parameters for these requirements may not be designed reasonably, and therefore it may not be possible to satisfy all requirements. Now, based on the given reward functions, reward and weight coefficients, and performance logs, you need to provide <num_spec> suggestions for modifying the weight coefficients.

<Env_description>

## Reward function, weights and logs

The specific code for the reward function is as follows:

<rew_function>

The weight coefficients and their corresponding performance results are given below:

<weight_training_result>

## Output Guide

Due to multiple input sets, it is necessary to inspect the performance changes resulting from variations in the weights, adjusting the weight values to achieve user goals as quickly as possible while ensuring a broad exploration range.

- **Weight adjustment and performance analysis**: The first input group is the starting point for the search, with all other input groups adjusted based on the first group. You need to output which weight variables corresponding to specific user requirement(s) the second and subsequent input groups have changed relative to the first set. Following this, analyze the impact of these adjustments on performance in term of direction and degree, both from the performance summarization and by directly referencing performance metrics (favoring the latter in case of conflicts).  Also, explain what these changes in metrics signify. Additionally, explain the characteristics in common of these input groups.

- **Listing possible reasons & Proposing solutions**: Analyze why the input groups may fail to meet user requirements. This may be caused by either relatively small weight values directly corresponding to certain requirement(s) or some requirement(s) being overly optimized. You need to analyze how to adjust each weight to better meet user needs. Firstly, analyze the direction in which a weight should be adjusted if necessary, increasing or decreasing the weight further when some requirements are not met. For fine-tuning adjustments, you can opt for intermediate weight values (as there are more than one weight value corresponding to an input) or maintain the weight at the starting point value, or align it with search strategy that a input group have already been performed. Additionally, analyze adjustment strategies, including cases where only one weight needs adjustment or when more than one weight requires adjustment, combining the adjusting strategies for these weights, following the principles presented below. Considering the principle of minimum weight adjustment, the number of weights modified relative to the starting point should generally be kept below half (in this case, no more than two weights) for each search group.

- **Adjustment step size**: The step size should increase to ensure a broad enough exploration range, or decrease to make refined search when necessary. Typically, larger step sizes such as 5x, 10x (or 1/5x, 1/10x for decreasing) and smaller step sizes like 1.5x, 2x (or 1/1.5x, 1/2x for decreasing) are common principles to follow. Some examples for determining the basic step size:
  -  When inputs are `w_xx=3` and `w_xx=15`, a specific user requirement (referred to as requirement A) is far from meeting performance requirements, then the search step size should be increased (5x, 10x). In scenarios where requirement A is well optimized but the remaining demands are not met, you should decrease the weight of `w_xx` and similarly adopt a large search step size (1/5x, 1/10x).
  -  During the final stages of the search when a more refined search is needed, for instance, when requirement A is poorly met at `w_xx=3` and over-optimized at `w_xx=15`, the value of `w_xx` should be more moderate.
- Human feedback: We provide an area below to put in feedback text written by humans, including the user's evaluation of the performance indicators and some improvement suggestions. If no feedback is given, the text will be set to `<Empty>`. You still need to analyze the performance indicators independently and come up with your modification strategies, and then combine them with human suggestions to appropriately allocate the output <num_spec> suggestions. Note that human feedback may be vague or incorrect, and you should specify and correct it according to the actual situation.

## Human feedback

<human_feedback>

'''

RWS1_example1 = '''
## Output Format Example

The following are examples of analysis and modification suggestions, incorporating the principles outlined above. Follow the format provided, and provide detailed corresponding analyses, **referencing performance metrics, user requirements, weight variables, numerical values as much as possible**, etc. (To avoid revealing suggestions, user requirements will be replaced with requirement A, requirement B, etc.)

### Input weight adjustment and Performance Analysis

- **Input Group 1 (Starting Point):** Requirement A shows little improvement during training, ...(analysis of other demands), while requirement C has been significantly optimized and performs exceptionally well. This may indicate that the requirement C has been excessively optimized.
- **Input Group 2:**  Compared to the starting point, the weight assigned to requirement has been increased from `w_a=3` to `w_a=15`, yet the performance of requirement A has not shown significant improvement. Other performance indicators remain largely consistent with the starting point.

(...analysis of the other three groups)

### Adjustment Strategy

Based on the analysis above, the reasons why the performance indicators are not being met in the input groups are:

- Since the improvements made in Input Groups 2-5 have been limited, the most likely reason is that the initial value of the weight `w_c` corresponding to requirement C is too high, thereby overshadowing the importance of other weights.
- Additionally, considering that requirement A and B have not been well optimized, it may be beneficial to individually increase their weight values. However, it has a lower priority.

Therefore, for the output groups, the following search strategies are proposed:

(NOTE THAT: (a). Do not contain any numerical value of adjusted weights, only show the direction for adjusting. (b)To facilitate pattern matching, each suggestion in the following output must occupy only one line and MUST contain a bold `**Suggestion X: **`. Previous output MUST NOT contain the string `**Suggestion`.)

- **Suggestion 1:** This output only attempts mutation(adjustment) from the starting point and solely decreases the weight corresponding to requirement C (mutation). Due to significant overshadowing, a step size of 1/5x is chosen.
- **Suggestion 2:** In addition to Suggestion 1, this output crosses the requirement to increase the weight of Demand A by 2.5 times (mutation + mutation&crossover). Since the adjustment range is hard to be determined, a mid-range step size of 2x is chosen.
- **Suggestion 3:** This suggestion only takes a 1/2x step size for the weight of requirement C compared to the starting point, while crossing the adjustment already made in input group 3 to increase the weight of Demand B (mutation + crossover), without further increase.
- **Suggestion 4:** This suggestion only further increasing the weight of requirement A (mutation). Due to the corresponding performance is far from satisfaction, a step size of 5x is chosen.
- **Suggestion 5:** This suggestion only further increases the weight of requirement B (mutation). Due to severe non-compliance, a step size of 5x is chosen.
'''

RWS1_example2 = '''
## Another Shortened Example

- Input Group 1 (Starting Point): Requirement A shows slight improvement, requirement B is moderately optimized but still falls short of the desired value, ...(analysis of other requirements).
- Input Group 2: Compared to the starting point, the weight assigned to requirement A has been increased from `w_a=3` to `w_a=15`, resulting in full satisfaction for requirement A, but a decrease in performance for requirement B compared to the starting point.
- Input Group 3: Compared to the starting point, the weight assigned to requirement B has been increased from `w_b=3` to `w_b=15`, resulting in full satisfaction for requirement B, but a decrease in performance for requirement A compared to the starting point.

(...analysis of the other two groups)

Based on the above analysis, the reasons for some performances not being met in the input groups is:

- Input Groups 2 and 3 attempt to optimize individual requirements, but due to conflicting nature of requirement A and requirement B, it may be beneficial to select a certain mid-point for the weight `w_a` corresponding to requirement A, or choose a mid-point for the weight `w_b` corresponding to requirement B, or combine these two strategies.

Therefore, the following strategies are proposed for the outputs:

**Suggestion 1:** A mid-point value may be chosen for the weight corresponding to requirement A, considering the weights from other Input Groups `w_a=3` and Input Group 2 `w_a=15`, with a 3x step size due to the 5x step size relative to the starting point in Input Group 2.
(strategies for other output groups)
'''