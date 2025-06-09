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
import sys
import os
import json
import re
from typing import Optional

from agent.model import Model
from agent.retriever import Retriever
from agent.tools import Tool
from agent.critic import Critic
from utils.execute import parse_code_from_string, python_repl_ast
from utils.utils import infer_dtype, get_df_info,extract_value_from_string,extract_action
from prompts import get_prompt
from collections import Counter
# global variables for python repl
import pandas as pd
import numpy as np
from datetime import datetime


class TableAgentSC:
    def __init__(
            self,
            model_name: str,
            retrieve_mode: str,
            embed_model_name: Optional[str] = None,
            task: str = 'tabfact',
            # agent_type: str = 'PyReAct',
            # agent_type: str = 'TableTools',
            agent_type: str = 'ReadSchema',
            top_k: int = 3,
            sr: int = 0,
            max_encode_cell: int = 10000,
            temperature: float = 0.8,
            top_p: float = 0.95,
            stop_tokens: Optional[list] = ['Observation:'],
            max_tokens: int = 1280,
            max_depth: int = 10,
            match="fuzzy",
            load_exist: bool = False,
            log_dir: Optional[str] = None,
            db_dir: Optional[str] = None,
            verbose: bool = False,
            sc:int=5,
    ):
        self.model = None
        self.model_name = model_name
        self.retrieve_mode = retrieve_mode
        self.embed_model_name = embed_model_name
        self.task = task
        self.agent_type = agent_type
        self.top_k = top_k
        self.sr = sr
        self.max_encode_cell = max_encode_cell
        self.max_depth = max_depth
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_exist = load_exist
        self.log_dir = log_dir
        self.db_dir = db_dir
        self.verbose = verbose
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.sc=sc
        self.model = Model(self.model_name)
        self.toolapi=Tool(self.model_name)
        self.criticapi=Critic(self.model_name)
        self.match=match
        self.retriever = Retriever(agent_type, retrieve_mode, embed_model_name, top_k=top_k, max_encode_cell=max_encode_cell, db_dir=db_dir, verbose=verbose)

    def is_terminal(self, text: str) -> bool:
        return 'final answer:' in text.lower()


    def query(self, prompt,df) -> str:

        response_text,token = self.model.query(
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
            n=self.sc,
        )
        self.total_input_token_count += token['prompt_tokens']
        self.total_output_token_count += token['completion_tokens']
        tools=[]
        print("====================")
        for i in range(self.sc):
            if response_text[i] is None:
                continue
            thought = response_text[i].split('Thought:')[-1].strip()
            if self.is_terminal(thought):
                final_answer = thought.split('Final Answer:')[-1].strip()
                # print("final_answer:",final_answer)
                tool=[final_answer]
                tools.append(tool)
            else:
                if 'Action:' not in response_text[i]:
                    print("Error: no Action provided.")
                    tool=[]
                    tools.append(tool)
                else:
                    parts = response_text[i].split('Action:')
                    first_action_after = 'Action:'.join(parts[1:]).strip()
                    action="Action:" + first_action_after
                    tool=extract_action(action)
                    if tool:
                        if ("def()" or "python") in tool[0]:
                            match = re.search(r"```(?:python)?\s*(.*?)\s*```", action, re.DOTALL)
                            if match:
                                code = match.group(1)
                                observation, memory ,flag= python_repl_ast(code, custom_locals={'df': df.copy()}, custom_globals=globals(), memory={})
                                if flag==0:
                                    tool=[]
                                else:
                                    tool=[observation]
                            else:
                                tool=[]
                    tools.append(tool)
                    print("extract_action:",tool)
        print("====================")
        # if len(final_answers) > 3:
        #     # 使用 Counter 统计每个 final_answer 出现的次数
        #     counter = Counter(final_answers)
        #     # 找到出现次数最多的 final_answer
        #     most_common_answer = counter.most_common(1)[0][0]
        #     # 找到最频繁出现的 final_answer 的第一个索引
        #     most_common_index = final_answers.index(most_common_answer)
        #     return response_text[most_common_index]
        # else:
        index=self.find_most_frequent_tool_index(tools)
        if index is not None:
            return response_text[index]
        else:
            return response_text[0]

    def find_most_frequent_tool_index(self,tools_list):
        # Step 1: 过滤掉空列表 []，并统计非空子列表中的 tool 频率
        valid_sublists = [sublist for sublist in tools_list if sublist]  # 跳过 []
        if not valid_sublists:  # 如果全是空列表，返回 None
            return None
        
        all_tools = [tool for sublist in valid_sublists for tool in sublist]
        tool_counter = Counter(all_tools)
        max_count = max(tool_counter.values())
        most_common_tools = [tool for tool, count in tool_counter.items() if count == max_count]
        
        # Step 2: 在原始列表中，找到包含高频 tool 的非空子列表
        candidate_indices = []
        for idx, sublist in enumerate(tools_list):
            if not sublist:  # 跳过空列表 []
                continue
            for tool in most_common_tools:
                if tool in sublist:
                    candidate_indices.append((idx, len(sublist)))
                    break  # 避免重复添加
        
        # Step 3: 按子列表长度和索引选择
        if not candidate_indices:
            return None
        
        candidate_indices.sort(key=lambda x: (-x[1], x[0]))  # 长度降序，索引升序
        return candidate_indices[0][0]


    # Solver loop follows the ReAct framework: https://github.com/ysymyth/ReAct.
    def solver_loop_tool(self, df: pd.DataFrame, question,caption,prompt: str) -> str:
        if self.verbose:
            print(prompt, end='')
        critic_cricle=0
        critic_python=0
        memory = {}
        aaa=0
        n_iter = self.max_depth
        solution = ''
        init_prompt = prompt
        action_stack=[]
        tool_count={"get_row":0,"get_value":0,"find_columns_containing_value_fuzzy":0,"find_column_format":0,"get_column_meaning":0,"def":0}
        for i in range(self.max_depth):
            entry={'thought': '', 'action': '', 'observation': ''}
            solution += 'Thought: ' # always start with thoughts            prompt = init_prompt + solution
            prompt = init_prompt + solution
            text = self.query(prompt,df).strip()
            # solution+= text
            
            if self.verbose:
                print('Thought: ' + text)

            # first check if it is terminal
            if self.is_terminal(text):
                thought = text.split('Final Answer:')[0].replace('Thought:', '').strip()
                answer=text.split('Answer:')[-1].split('\n')[0].strip()
                n_iter = i + 1
                solution+=text
                break
            
            if 'Action:' not in text:
                observation = 'Error: no Action provided.'
                solution+= text
            else:
                thought = text.split('Action:')[0].replace('Thought:', '').strip()
                parts = text.split('Action:')
                first_action_after = 'Action:'.join(parts[1:]).strip()
                action="Action:" + first_action_after
                
                tool=extract_action(action)
                print("extract_action:",tool)
                print(action_stack)
                while not route_check(action_stack,tool):
                # while 0:
                    if critic_cricle>5:
                        # observation = 'You have used the same tool too many times. You should reconsoider your action.'
                        break
                    critic_cricle+=1
                    text,thought,actoin=self.criticapi.critic_circle(init_prompt+solution,tool)
                    tool=extract_action(actoin)
                    if self.is_terminal(text):
                        answer=text.split('Answer:')[-1].split('\n')[0].strip()
                        n_iter = i + 1
                        solution+=text
                        aaa=1
                        break  
                if aaa:
                    break        
                    
                if tool==[]:
                    observation = 'Error: no valid Action provided.  You can only use a same tool in one action or you strictly follow the format.'
                elif 'get_column_meaning' in tool[0] :
                    tool_count["get_column_meaning"]+=1
                    # 可以根据thought对于对应的列进行检查
                    response=self.toolapi.get_table_description(df,thought,question)
                    observation=response
                elif  'get_value' in tool[0] :
                    
                    observation=""
                    for t in tool:
                        tool_count["get_value"]+=1
                        extract_tool=extract_value_from_string(t)

                        if extract_tool:
                            response=self.toolapi.find_columns_containing_value(df, extract_tool)
                            observation += response
                            observation += '\n'
                        else:
                            extract_tool= ''
                            observation="Please strictly follow the format. Or if you find a blank value, you can use def() to find the blank value."

                elif 'get_row'in tool[0]:
                    observation=""
                    for t in tool:
                        extract_tool=extract_value_from_string(t)
                        # text=re.search(r"get_row\(['\"]([^'\"]+)['\"]\)", tool)
                        if extract_tool:
                            # text= text.group(1).strip()
                            # print(text)
                            tool_count["get_row"]+=1
                            if 'df' in text:
                                observation="Error: you should use the get_row() with the format of get_row(\"index\"), only pass a string representing the row index.."
                            else:    
                                response=self.toolapi.get_row(df, extract_tool)
                                observation +=response
                                observation += '\n'
                        else:
                            extract_tool= ''
                            observation="please strictly follow the format.you should reuse it of get_row(\"value\")"
                elif  'find_columns_containing_value_fuzzy' in tool[0]:
                    observation=""
                    for t in tool:
                        extract_tool=extract_value_from_string(t)
                        if extract_tool:

                            tool_count["find_columns_containing_value_fuzzy"]+=1
                            if self.match=="fuzzy":
                                response=self.toolapi.optimized_fuzzy_search(df,extract_tool)
                            else:
                                response=self.toolapi.find_columns_containing_value_fuzzy(df,caption,extract_tool)
                            observation+=response
                            observation += '\n\n'
                        else:
                            extract_tool= ''
                            observation="please strictly follow the format.you should reuse it of find_columns_containing_value_fuzzy(\"value\")"
                    
                elif 'find_column_format' in tool[0]:
                    observation=""
                    for t in tool:  
                        extract_tool=extract_value_from_string(t)
                        tool_count["find_column_format"]+=1
                        response,iter_count=self.toolapi.find_column_format(df, extract_tool , thought,self.criticapi)
                        print(iter_count)
                        critic_python=iter_count+critic_python
                        observation+=response
                        observation += '\n'
                elif ('python' or 'df' or 'def' ) in tool[0]:
                    print("tool:",tool)
                      # 使用正则表达式提取括号中的代码，假设你想提取 python_repl_ast(...) 中的内容
                    # 可能匹配不到，是不是需要使用llm处理代码
                    # 只在前后缀匹配时执行切片
                    match = re.search(r"```(?:python)?\s*(.*?)\s*```", action, re.DOTALL)
                    if match:
                            code = match.group(1)

                    else:
                        if "df" in tool[0]:
                            code = tool[0]
                        else:
                            code = ''

                    print("code=", code)
                    
                    if code== '':
                        observation = 'Error: no code provided. you should check the code is print with the format of python_repl_ast(code)'
                    else: 
                        tool_count["def"]+=1
                        # execute the code, we need to pass the dataframe, and pandas as pd, numpy as np to the locals
                        code = parse_code_from_string(code)
                        critic_python_0=0
                        while 1:
                            observation, memory ,flag= python_repl_ast(code, custom_locals={'df': df}, custom_globals=globals(), memory=memory)
                            if flag==0:
                                input=init_prompt+solution+f"{thought}\nAction:def()```{code}```\n"
                                code= self.criticapi.critic_python(input,observation)
                                critic_python_0+=1
                                if critic_python_0>3:
                                    observation +='\n Cannot critic the code, it\'s better to just start over with a fresh approach rather than keep tweaking the existing code. '
                                    break
                                continue
                            # if isinstance(observation, str) and self.model.get_token_count(observation) > self.model.context_limit:
                            #     observation = 'Observation is too long, we cannot query the API.'
                            #     break
                            if isinstance(observation, str) and observation != '':
                                observation = observation.strip()
                                break
                            if observation == '':
                                observation = 'exexute code successfully, and nothing is printed'
                                break
                        critic_python+=critic_python_0
                        # observation, memory ,flag= python_repl_ast(code, custom_locals={'df': df}, custom_globals=globals(), memory=memory)
    
                        #     # if isinstance(observation, str) and self.model.get_token_count(observation) > self.model.context_limit:
                        #     #     observation = 'Observation is too long, we cannot query the API.'
                        #     #     break
                        # if isinstance(observation, str) and observation != '':
                        #     observation = observation.strip()
                        #     # break
                        # if observation == '':
                        #     observation = 'exexute code successfully, and nothing is printed'
                        #     # break
                        # critic_python+=critic_python_0
                    
                else:
                    if 'df' in tool:
                        observation="Error: you should check the code is print with the format of def()```python\n```"
                    observation = 'Error: no valid action provided. you should strictiy follow the tool description and the format.'
                solution+= f"{thought}\n{action}\n"
            # if observation has multiple lines, we need to add new line at the beginning
            if '\n' in str(observation):
                observation = '\n' + str(observation)

            solution += f'\nObservation: {observation}\n'

            if self.verbose:
                print(f'Observation: {observation}')
            entry["observation"] = observation

        answer = text.split('Answer:')[-1].split('\n')[0].strip()
        print(action_stack)
        tool_count["critic_circle"]=critic_cricle
        tool_count["critic_python"]=critic_python
        return answer, n_iter, solution, tool_count
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        return get_prompt(self.task, self.agent_type, prompt_type, **kwargs)

    def run(self, data:dict, sc_id: int = 0) -> dict:
        log_path = os.path.join(self.log_dir, 'log', f'{data["id"]}-{sc_id}.json')
        # Load the log file if it exists
        if os.path.exists(log_path) and self.load_exist:
            with open(log_path) as fp:
                result = json.load(fp)
            return result


        if self.verbose:
            print('=' * 25 + f' {data["id"]} ' + '=' * 25)

        # Read table
        table_caption = data.get('table_caption', '')
        query = data['statement'] if 'statement' in data else data['question']
        if self.task=="bird":
            path=data['table_text']
            print(path)
            with open(path, "r", encoding="utf-8") as tf:
                
                table_text = json.load(tf)  # 统一转小写方便查找
        # df = pd.DataFrame(table_text[1:], columns=table_text[0])
        else:
            table_text=data['table_text']
        df = pd.DataFrame(table_text[1:], columns=[col.replace("\n", " ").replace("\\n", " ") for col in table_text[0]])
        # 处理重复的列名
        def rename_duplicate_columns(df):
            # 获取列名
            cols = pd.Series(df.columns)
            
            # 标记重复列名的索引
            for i, col in enumerate(cols):
                if cols.tolist().count(col) > 1:  # 如果列名重复
                    # 给重复列名加上后缀，如 'name1', 'name2'
                    cols[i] = f"{col}{cols.tolist().count(col)}"
            
            # 重新分配列名
            df.columns = cols
            return df

        # 调用函数处理重复列名
        df = rename_duplicate_columns(df)

        if (self.agent_type == 'PyReAct' and 3 * df.shape[0] * df.shape[1] > self.model.context_limit) or (self.agent_type == 'RandSampling' and 3 * self.top_k * df.shape[1] > self.model.context_limit):
            prompt = ''
            answer = solution = 'Error: table is too large.'
            n_iter = init_prompt_token_count = 0
            if self.verbose:
                print('Error: table is too large.')
        else:
            df = infer_dtype(df)
            table_markdown = get_df_info(df)

            if self.task=="bird":
                prompt = self.get_prompt('solve_table_prompt', table_caption=table_caption, query=query, table=table_markdown,evidence=data["evidence"])
            else:
                prompt = self.get_prompt('solve_table_prompt', table_caption=table_caption, query=query, table=table_markdown)
            
            answer, n_iter, solution ,tool_count= self.solver_loop_tool(df,query,data["table_caption"],prompt)
            # print(self.toolapi.total_iutput_token_count)
            
            input_token=self.total_input_token_count+self.toolapi.total_input+self.criticapi.total_critic_input
            output_token=self.total_output_token_count+self.toolapi.total_output+self.criticapi.total_critic_input
            print("input_token=",input_token)
            print("output_token=",output_token)
        result = {
            'id': data['id'],
            'sc_id': sc_id,
            'table_caption': table_caption,
            'query': query,
            'solution': solution,
            'answer': answer,
            'label': data['label'],
            'n_iter': n_iter,
            # 'init_prompt_token_count': init_prompt_token_count,
            'total_token_count': input_token + output_token,
            'input_token_count': input_token,
            'output_token_count': output_token,
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
            'tool_count': tool_count,
        }
        if 'orig_id' in data:
            result['orig_id'] = data['orig_id']

        with open(log_path, 'w') as fp:
            json.dump(result, fp, indent=4)
        with open(log_path.replace('.json', '.txt'), 'w',encoding="utf-8") as fp:
            fp.write(prompt + solution)

        return result

import re


def route_check(action_stack,tool):
    # 检查当前工具是否在栈中
    if len(action_stack) == 0:
        for t in tool:
            action_stack.append(t)
        return True
    else:
        # 检查当前工具是否在栈中
        for t in tool:
            if t in action_stack:
                return False
            else:
                action_stack.append(t)
        return True
    

if __name__ == '__main__':
    print("This is a test")