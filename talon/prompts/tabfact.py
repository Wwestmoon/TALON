# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
pyreact_solve_table_prompt_tools = '''
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use tools to verify the statement.
- `get_column_meaning()`: This tool can help you understand the most relevant columns and column meanings about the question.
- `find_columns_containing_value_fuzzy(value)`: If you are searching for a specific value that differs from the column name, use this tool to identify potential variations of the value across columns. This is useful when the value may have different representations or may not match exactly with what is provided in the problem.
- `find_column_format(column)`: Find or Process the format of a specific column. This is useful when you need to find or process the value of a column.
- `get_value(value)`: Search for a specific value. This will help us determine if the value exists within a column.
- `get_row(row_index)`: Use this tool to retrieve a specific row from the table. Or you can use this tool to understand the relationshio of two colums in a row. Be carful to use it until you exact know the index of the row, do not assume the index of the row. 
- `def()`: This tool can generate Python code,you should generate with the format of def()```python\n[### the code you generate ###]\n``` to fetch data or calculate or other actions. If you define a variable, you must print it to show the output.Do not output any additional text. Ensure that the code is self-contained and executable. Do not sample values.

Strictly follow the given format to respond:
Thought: you should always print about the thinking process about what to do based on the previous observation.
Action: Directly output the tool you choosed to use, do not output other inforamtion.
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations

Final Answer: the final answer to the original input question, only print the answer,the answer should not be a variable, no other form.

Notes:

- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: true/false".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.
- You can call a same tool multiple times simultaneously. For example, you can output Action: tool1(args1), tool1(args2) at one same time.
- If you use def() tool, Do not sample data, just process the dataframe. 
- You have to make sure gather enough information to answer the question. 

Now, given a table regarding "{table_caption}", please verify the statement: "{query}".

{table}

Begin!
'''


