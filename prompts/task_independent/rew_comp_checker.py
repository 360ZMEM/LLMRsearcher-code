# A variant of Training Log Analyzer (TLA)
RCC_str = """
We employ reinforcement learning (RL) to train agents by developing reward functions that align with user requirements, which include specific numerical performance demands. Given the training logs for each performance indicator related to these user requirements, you are required to compare the results of each performance indicator against the corresponding user requirement targets individually, and output the judgment (`[YES]` or `[NO]`) regarding compliance with the user requirements.

## Objectives

Below are some of the user requirements that must be satisfied, which may encompass one or more performance metrics. Please note that the requirements are presented in a Python-formatted `dict`, where the key represents the name of the user requirement and the value denotes the specific description of the requirement.

```python
<objectives>
```

## Performance Results

Below is the performance log that requires analysis; [HGH] indicates that higher values are preferable, while [LOW] indicates that lower values are preferable.

<comp_perf>

## Output Format Requirements

The output consists of the following sections:

### Analyze user requirements and corresponding performance metrics

Output the changes in performance metrics for each user requirement individually, and assess whether they meet the user's needs. Your output should include: (a) the initial metric value when training began; (b) the ratio of the final five values relative to the starting point; (c) the numerical demands of user requirements, which must be answered with either YES or NO in comparison to the performance outcome; (d) the standard deviation information can be combined to assist judgment.

### Formatted output

After analyzing each user requirement, provide a structured summary indicating whether the performance objectives for all user requirements have been met. The output is a Python-formatted `dict` similar to the input, with keys consistent with the input. The corresponding value under each key should reflect the earlier judgment on whether each demand has been met. The value must be a "[YES]" string when the answer is "YES", and "[NO]" when the answer is "NO", and should be output in a Python code block formatted as `python`.
"""

RCC_str_example = """
## Format Example

### Input

If the user requirements are as follows, where `foo` and `bar` serve as examples of requirements with numerical objectives, while `qux` represents examples of performance metrics (which are of relatively low priority):

```python
{'foo':'The number of xxx should be reduced to be zero.','bar':'The number of xxx should be reduced to be zero.','qux':'This value should be reduced.'}
```

If the training logs are as follows, where [HGH] indicates that higher values are preferable, while [LOW] indicates that lower values are preferable.

[LOW]AVG of performance metric of foo: [8.36, 24.72, 1.96, 0.92, 0.56, 7.84, 0.84, 1.52, 0.44, 0.96, 0.33]

[LOW]STD of performance metric of foo: [18.317, 41.542, 4.152, 3.298, 2.368, 21.49, 3.916, 4.527, 1.791, 1.793, 1.196] 

[LOW]AVG of performance metric of bar: [380.96, 52.1, 7.36, 23.8, 13.32, 120.64, 125.36, 42.4, 216.72, 143.6, 85.54, 85.78, 67.78, 42.1]

[LOW]STD of performance metric of bar: [82.466, 45.539, 18.483, 50.931, 29.259, 100.173, 148.53, 38.722, 148.54, 99.713, 98.323, 72.813, 75.733, 57.322]

[LOW]AVG of performance metric of qux: [96.957, 76.588, 56.087, 53.268, 52.556, 57.549, 57.911, 53.582, 55.985, 55.302, 54.883]

[LOW]STD of performance metric of qux: [3.429, 8.702, 2.568, 2.591, 1.462, 4.969, 7.021, 1.874, 3.437, 3.482, 2.94] 

### Example output

- Requirement 'foo': The initial value is 8.36, and the average of the last five values decreases to approximately 0.5 (~5%). Since the user requirement stipulates that this value must be reduced to zero, it is evident that the requirement is essentially satisfied.
- Requirement 'bar': The initial value is 380.9, and the average of the last five values decreases to approximately 80 (~20%). However, there remains a significant gap between this outcome and the user requirement that demands this value must be reduced to zero; therefore, the requirement is not satisfied.
- Requirement 'qux': The initial value is 96.9, and the average of the last five values decreases to approximately 55 (~60%). Since the requirement stipulates that this value must be reduced and we have observed a significant decrease, the requirement is thus satisfied.

Finally, the output formatted as a Python dictionary is as follows:

```python
{'foo':'[YES]','bar':'[NO]','qux':'[YES]'}
```
"""
