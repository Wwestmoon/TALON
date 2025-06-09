import json
import os
import re
import sys
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor
from torch import cosine_similarity
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)
from utils.execute import parse_code_from_string, python_repl_ast
from utils.load_data import load_dataset
import pandas as pd
import numpy as np
from datetime import datetime
# from model import Model
from agent.model import Model
import random

from typing import Optional
import pandas as pd
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def critic_length(self,trajectory):
        """
        处理路由信息，进行自我修正。
        
        返回：
        - 修正后的路由信息字符串
        """

        prompt = """
        You have exceeded the maximum of 8 execution steps. Based on previous observations, you must now:

        1. Re-evaluate the problem-table relationship
        2. Synthesize lessons learned from prior attempts
        3. Develop an alternative solution approach

        Key Requirements:
        - Leverage previously used tools effectively
        - Avoid redundant tool usage
        - Create a phased implementation plan
        - Generate sequential, non-repetitive actions

        Output Format:
        thought1: [Revised thought process]
        action1: [Precise diagnostic action]

        Attached is your original execution trace for reference: {original_trace}
        """
        print(prompt)
        # print("-----------------------------------------------")
        response=self.query(prompt)
        print(response)


def critic_action(self,trajectory,tool):
        n=2
        # check the action if meet the circle
        length= len(trajectory)
        for i in range(length):
            if trajectory[i]["action"]==tool:
                if length-i>n:
                    prompt=f"""you have already executed the tool several steps ago,you should reconsider the trajectory and make a plan about how to solve the question.
here is the trajectory:
{trajectory}    
Based on the trajectory, you should determine whether the thought was right based on the perception, if it was wrong, you should tell me the reason why it was wrong
    """
                    # print(prompt)
                    response=self.query(prompt).strip()
                    thought = response.split('Action:')[0].replace('Thought:', '').strip()
                    tool = response.split('Action:')[-1].strip()
                    return thought,tool

def critic_long_iter(trajectory):
    """
    检查迭代次数是否过长
    """
    length = len(trajectory)
    if length > 10:
        prompt = f"""You have executed too many steps,you should make a plan about how to solve the question based on the before trajectory and avoid the same situation."""
class Critic:
    
    def __init__(
        self,
        model_name: str,
        # retrieve_mode: str,
        # embed_model_name: Optional[str] = None,
        # task: str = 'tabfact',
        # agent_type: str = 'PyReAct',
        # agent_type: str = 'TableTools',
        top_k: int = 3,
        sr: int = 0,
        max_encode_cell: int = 10000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_tokens: Optional[list] = ['Observation:'],
        max_tokens: int = 512,
        max_depth: int = 5,
        # load_exist: bool = False,
        # log_dir: Optional[str] = None,
        # db_dir: Optional[str] = None,
        # verbose: bool = False,
        ):
        
        self.model = None
        self.model_name = model_name
        self.retrieve_mode = 'embed'
        self.embed_model_name = "text-embedding-3-large"
        # self.task = task
        # self.agent_type = agent_type
        self.top_k = top_k
        self.sr = sr
        self.max_encode_cell = max_encode_cell
        self.max_depth = max_depth
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_critic_input = 0
        self.total_critic_output = 0
        self.model = Model(self.model_name)
        # self.retriever = Retriever(agent_type, retrieve_mode, embed_model_name, top_k=top_k, max_encode_cell=max_encode_cell, db_dir=db_dir, verbose=verbose)

    def query(self, prompt) -> str:
        # input_token_count = self.model.get_token_count(prompt)
        # if input_token_count > self.model.context_limit:
        #     return f'Prompt length -- {input_token_count} is too long, we cannot query the API.'
        response_text, token = self.model.query(
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
            n=1,
        )
        self.total_critic_input += token['prompt_tokens']
        self.total_critic_output += token['completion_tokens']

        return response_text[0]



    def critic_python_tool(self,python_code,error,sampled_values,thought,column_dtype):
        """
        检查错误的代码
        """
        print("=================================")
        print("critic python code!!!!!! for tool ")
        print("===================================")
        prompt = f"""You have executed a python code when processing the format of the column, 
        the thought of process the column is: 
        {thought},
        the sample values of the column are {sampled_values}, 
        and the code is:
        {python_code}, 
        but it has an error, here is an error message: 
        {error}

        Analyze the error message to:

        1. Diagnose the root cause of the failure .
        2. Validate the code logic against the provided DataFrame (df) based on the sample values.
        3. Formulate a corrected code, you can generate different code to process the column based on the thought and the sample values to avoid such an error.
        4. If Sample data of the table, you should modify the code to avoid sampling data.

        Critical Constraints:
        - The table is already available as a pandas DataFrame (df)
        - Never sample the table data during analysis.

        please think step by step and output the correct code with the format of ```python
        {python_code}```.
        """
        response = self.query(prompt)
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else None
        # print(prompt)
        print("critic: ", extracted_code)
        return extracted_code
    
    def critic_circle(self,trajectory,action):
        """
        处理路由信息，进行自我修正。
        
        返回：
        - 修正后的路由信息字符串
        """
        prompt = f"""As an AI assistant analyzing tabular data, I've noticed you've repeatedly called the same data processing tool multiple times. Based on previous observations, please:

            1. Re-evaluate your current approach
            2. Generate a new 'thought' process that considers:
            - Efficiency improvements
            - Avoiding redundant operations including all the history actions
            - Leveraging previously obtained results from the trajectory
            3. Provide a revised 'action' plan that:
            - avoid duplicate tool calls
            - generate a tool or final answer based on the trajectory


            Format your response with clear "Thought:" and "Action:" sections, ensuring the solution is both technically sound and resource-efficient.
             Output Format:
            Thought: Only print about the thinking process about what to do next .
            Action: Directly output the tool you choosed to use, do not output other inforamtion. 
        
            Here is the trajectory: {trajectory}   
            Here is the repeated action: {action}   
           

        """
        print(prompt)
        # print("-----------------------------------------------")
        text=self.query(prompt)
        # thought = response.split('Action:')[0].replace('Thought:', '').strip()
        thought = text.split('Action:')[0].replace('Thought:', '').strip()
        parts = text.split('Action:')
        first_action_after = 'Action:'.join(parts[1:]).strip()
        action="Action:" + first_action_after
        return text,thought,action

    def critic_python(self, trajectory,error):
        """
        检查python代码的正确性
        """

        prompt=f"""
        You have to read the table and the query, and all the thinking process, to identity the root cause of the error.
        Analyze the execution trace and error message to:

        1. Diagnose the root cause of the failure based on the error message and the trajectory.
        2. Validate the code logic against the provided DataFrame (df) based on the observations.
        3. Formulate a corrected code.
        4. If Sample data of the table, you should modify the code to avoid sampling data.

        Critical Constraints:
        - The table is already available as a pandas DataFrame (df)
        - Never sample the table data during analysis.

        Please Note: You should directly output the revised python code with the format of ```python\n``` to avoid such an error,no other additional text.


        Here is the thinking process: 
        {trajectory}   
        Here is the error message: 
        {error}

        """
        print(prompt)
        # print("-----------------------------------------------")
        response=self.query(prompt)
        print(response)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            code = match.group(1)
            print(code)
            return code
        else:
            return None
    

    def critic_length(self,trajectory):
        """
        处理路由信息，进行自我修正。
        
        返回：
        - 修正后的路由信息字符串
        """

        prompt = f"""
        You have exceeded the maximum of 8 execution steps. Based on previous observations, you must now:

        1. Re-evaluate the problem-table relationship
        2. Synthesize lessons learned from prior attempts
        3. Develop an alternative solution approach

        Key Requirements:
        - Leverage previously used tools effectively
        - Avoid redundant tool usage
        - Create a phased implementation plan
        - Generate sequential, non-repetitive actions

        Output Format:
        thought1: [Revised thought process]
        action1: [Precise diagnostic action]

        Attached is your original execution trace for reference: {trajectory}
        """
        print(prompt)
        # print("-----------------------------------------------")
        text=self.query(prompt)
        # thought = response.split('Action:')[0].replace('Thought:', '').strip()
        thought = text.split('Action:')[0].replace('Thought:', '').strip()
        parts = text.split('Action:')
        first_action_after = 'Action:'.join(parts[1:]).strip()
        action="Action:" + first_action_after
        return thought,action



