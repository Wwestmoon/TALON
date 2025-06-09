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

import io
import csv
import json
import warnings
import unicodedata
import re
import numpy as np
import pandas as pd

from utils.execute import parse_code_from_string


def read_json(text):
    res = parse_code_from_string(text)
    return json.loads(res)


def is_numeric(s):
    try:
        float(s)
    except:
        return False
    return True


def table_text_to_df(table_text):
    df = pd.DataFrame(table_text[1:], columns=table_text[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)
    return df


def infer_dtype(df):
    """
    Attempt to convert columns in a DataFrame to a more appropriate data type.

    :param df: Input DataFrame
    :return: DataFrame with updated dtypes
    """

    for col in df.columns:
        try:
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')

            # If the column type is still object (string) after trying numeric conversion, try datetime conversion
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='raise')
        except:
            pass

    return df


def get_df_info(df):
    buf = io.StringIO()
    df.info(verbose=True, buf=buf)
    return buf.getvalue()


def to_partial_markdown(df, n_visible):
    df = df.astype('object')
    df = df.fillna(np.nan)
    if n_visible == -1:
        return df.to_markdown(index=False)
    if n_visible == 0:
        return ''
    skip_rows = n_visible < df.shape[0]
    skip_cols = n_visible < df.shape[1]
    n_visible //= 2

    if skip_cols:
        new_df = df.iloc[:,:n_visible]
        new_df.loc[:,'...'] = '...'
        new_df = pd.concat([new_df, df.iloc[:,-n_visible:]], axis=1)
    else:
        new_df = df

    if skip_rows:
        rows = new_df.to_markdown(index=False).split('\n')
        row_texts = rows[1].split('|')
        new_row_texts = ['']
        for text in row_texts[1:-1]:
            if text[0] == ':':
                new_text = ' ...' + ' ' * (len(text) - 4)
            else:
                new_text = ' ' * (len(text) - 4) + '... '
            new_row_texts.append(new_text)
        new_row_texts.append('')
        new_row = '|'.join(new_row_texts)
        output = '\n'.join(rows[:2 + n_visible] + [new_row] + rows[-n_visible:])
    else:
        output = new_df.to_markdown(index=False)
    return output


def markdown_to_df(markdown_string):
    """
    Parse a markdown table to a pandas dataframe.

    Parameters:
    markdown_string (str): The markdown table string.

    Returns:
    pd.DataFrame: The parsed markdown table as a pandas dataframe.
    """

    # Split the markdown string into lines
    lines = markdown_string.strip().split("\n")

    # strip leading/trailing '|'
    lines = [line.strip('|') for line in lines]

    # Check if the markdown string is empty or only contains the header and delimiter
    if len(lines) < 2:
        raise ValueError("Markdown string should contain at least a header, delimiter and one data row.")

    # Check if the markdown string contains the correct delimiter for a table
    if not set(lines[1].strip()) <= set(['-', '|', ' ', ':']):
        # means the second line is not a delimiter line
        # we do nothing
        pass
    # Remove the delimiter line
    else:
        del lines[1]

    # Replace '|' in the cells with ';'
    stripe_pos = [i for i, c in enumerate(lines[0]) if c == '|']
    lines = [lines[0]] + [line.replace('|', ';') for line in lines[1:]]
    for i in range(1, len(lines)):
        for j in stripe_pos:
            lines[i] = lines[i][:j] + '|' + lines[i][j+1:]

    # Join the lines back into a single string, and use StringIO to make it file-like
    markdown_file_like = io.StringIO("\n".join(lines))

    # Use pandas to read the "file", assuming the first row is the header and the separator is '|'
    df = pd.read_csv(markdown_file_like, sep='|', skipinitialspace=True, quoting=csv.QUOTE_NONE)

    # Strip whitespace from column names and values
    df.columns = df.columns.str.strip()

    # Remove index column
    df = df.drop(columns='Unnamed: 0')

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # normalize unicode characters
    df = df.map(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)

    return df

def extract_value_from_string(s):
    """
    从字符串中提取括号内的值，支持各种类型的内容，包括数字、字符串、表达式等。
    :param s: 包含括号的字符串，例如 'get_row(13)', 'get_row("13")', 'get_row(\'Hello\')', 'get_row(3.14)'
    :return: 提取出的括号内的值，原样返回
    """
    
    match = re.search(r'\((.*)\)', s, re.DOTALL)  # 查找括号内的内容
            # re.search(r'\(([^)]+)\)', s)
    if match:
        value = match.group(1)  # 获取括号内的内容
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value  # 返回原始的内容，不做任何转换
    else:
        raise ValueError("No valid value found inside parentheses.")
    

def smart_split_actions(action_str):
    parts = []
    bracket_level = 0
    current = ''
    in_quotes = False

    for char in action_str:
        if char == '"' or char == "'":
            in_quotes = not in_quotes
        elif char == '(' and not in_quotes:
            bracket_level += 1
        elif char == ')' and not in_quotes:
            bracket_level -= 1
        elif char == ',' and bracket_level == 0 and not in_quotes:
            parts.append(current.strip())
            current = ''
            continue
        current += char

    if current:
        parts.append(current.strip())

    return parts

def extract_action(response):
    # 使用正则表达式提取所有 Action
    if "python " in response.lower() or "def()" in response.lower():
        action=response.split("Action:")[-1].strip()
        print(f"Extracted python code")
        return [action]
    # 使用正则表达式提取所有 Action
    actions = re.findall(r'Action:\s*(.*)', response)
    print(f"Extracted actions: {actions}")
    
    if not actions:
        return "No actions detected."
    
    tool_info = []
    for action in actions:
        action = action.strip()
        if action.startswith("```python"):
            # 如果是python代码块，提取整个代码
            code_block = re.search(r'```python(.*?)```', response, re.DOTALL)
            if code_block:
                tool_info.append(("python_code", code_block.group(1).strip()))
            else:
                tool_info.append(("python_code", None))
        else:
            # 正常处理工具调用
            if ',' in action:
                sub_actions = smart_split_actions(action)
                for sub_action in sub_actions:
                    match = re.match(r'(\w+)\((.*)\)', sub_action)
                    if match:
                        tool_name = match.group(1)
                        tool_value = match.group(2)
                    else:
                        tool_name = sub_action.strip()
                        tool_value = None
                    tool_info.append((tool_name, tool_value))
            else:
                match = re.match(r'(\w+)\((.*)\)', action)
                if match:
                    tool_name = match.group(1)
                    tool_value = match.group(2)
                else:
                    tool_name = action.strip()
                    tool_value = None
                tool_info.append((tool_name, tool_value))
    
    # 如果只有一种工具
    tool_names = [info[0] for info in tool_info]
    if len(set(tool_names)) == 1:
        tool_values = [f"{tool_name}({tool_value})" if tool_name != "python_code" else f"```python\n{tool_value}\n```" for tool_name, tool_value in tool_info]
        # print(f"Extracted tool values: {tool_values}")
        return tool_values
    else:
        return []