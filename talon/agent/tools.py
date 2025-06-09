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
# from relation_filter import SentenceTransformerRetrieval
# from critic import Critic
from agent.model import Model
from agent.relation_filter import SentenceTransformerRetrieval
from agent.critic import Critic
import random

from typing import Optional
import pandas as pd
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity




from fuzzywuzzy import fuzz

def fuzzy_similarity(str1, str2):
    return fuzz.ratio(str1, str2) / 100.0

class Tool:
    
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
        self.total_input = 0
        self.total_output = 0
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
        self.total_input += token['prompt_tokens']
        self.total_output += token['completion_tokens']

        return response_text[0]
    
        
    def find_columns_containing_value(self, df, search_value):
        """
        在 Pandas DataFrame 中查找包含特定值的列。
        
        参数：
        - df: pandas DataFrame
        - search_value: 要查找的值
        
        返回：
        - 包含该值的列名列表
        """
        matched_columns = []
        
        # for col in df.columns:
        #     if search_value.lower() == col.lower():
        #             return "The value you are searching for is a column name. This is not the intended usage of get_value."

        #     if search_value in df[col].astype(str).values:  # 转换为字符串匹配
        #         matching_indices = df.index[df[col].astype(str) == search_value].tolist()
        #         matched_columns.append((col, matching_indices))
        for col in df.columns:
            if search_value.lower() == col.lower():
                return "The value you are searching for is a column name. This is not the intended usage of get_value."
            for value in df[col].astype(str).values:
                    # 检查是否为数字
                if value.isdigit():
                    #如果列值是数字，进行精确匹配
                    if value == search_value:
                        matching_indices = df.index[df[col].astype(str) == search_value].tolist()
                        matched_columns.append((col, matching_indices,value))
                        break  # 找到匹配后跳出当前列的循环
                else:
                    #     # 如果列值不是数字，进行部分匹配
                    if search_value.lower() in value.lower():
                        escaped_search_value = re.escape(value)
                        matching_indices = df.index[df[col].astype(str).str.contains(escaped_search_value),].tolist()
                        matched_columns.append((col, matching_indices,value))
                        break  # 找到匹配后跳出当前列的循环

        if len(matched_columns) > 0:
            rtn_str = f"The columns below contain the value {search_value}:"
            for column in matched_columns:
                if len(column[1])<=5:
                    # 如果匹配的行数小于等于5，直接输出行索引
                    rtn_str += f"\n-[Value]: {column[2]} -[Column Name]: {column[0]} -[row_indices]: {column[1]}"
                else:
                    # 如果匹配的行数大于5，随机抽取5个行索引输出
                    random_indices = random.sample(column[1], 5)
                    rtn_str += f"\n-[Value]: {column[2]} -[Column Name]: {column[0]} -[sample_row_indices]: {random_indices}, and please note there are {len(column[1])} indexs in total that contain the value. you should use the tool of def() to get all of the indexes."
                # rtn_str += f"\n-[Value]: {column[2]} -[Column Name]: {column[0]} -[row_indices]: {column[1]}"
        else:
            rtn_str = f"No columns contain the exact value. You may try the tool of find_columns_containing_value_fuzzy(\"{search_value}\")."

        return rtn_str
    
    def get_row(self, df, row_index):
        """
        获取 Pandas DataFrame 中指定行的内容。
        
        参数：
        - df: pandas DataFrame
        - row_index: 行索引
        
        返回：
        - 指定行的内容
        """
        if row_index.isdigit():
            row_index = int(row_index)
        else:
            return "Row index must be an integer. You can directly use the tool of get_row() to get the row index, no other additional ouptput."
        if row_index < 0 or row_index >= len(df):
            return "Row index out of range."
        
        row_content = df.iloc[row_index].to_dict()
        return f"Row {row_index} content: {row_content}"
    
    def find_columns_containing_value_fuzzy(self, df ,source,search_value):
        # print(f"Searching for '{search_value}' in the dataframe...")
        # Initialize variables
        value_map = defaultdict(lambda: [])
        values = set()

        # Prepare the search value
        search_value = str(search_value).strip()
        print(f"Searching for '{search_value}' in the dataframe...")
        for col in df.columns:
            # print(col)
            if search_value.lower() == col.lower():
                return f"The '{search_value}' is the column of the table."
        # Loop through each column in the dataframe
        for col in df.columns:
            # print(type(df[col]))
            # print(df[col])
            # Get unique values in the column
            distinct_values = df[col].dropna().astype(str).unique()
            
            # Map each value to its corresponding column
            for value in distinct_values:
                value_map[value].append(col)
                values.add(value)
                
        import pickle

        embeddings = OpenAIEmbeddings(model=self.embed_model_name)
        # 缓存嵌入向量
        cache_file = f"embeddings/{source}.pkl"
        try:
            with open(cache_file, "rb") as f:
                value_embeddings = pickle.load(f)
    
        except FileNotFoundError:
            all_values = list(values)
            value_embeddings = embeddings.embed_documents(all_values)
            with open(cache_file, "wb") as f:
                pickle.dump(value_embeddings, f)
        # Use OpenAI embeddings for encoding

        
        #Get embeddings for all unique values in the dataframe
        value_embeddings = embeddings.embed_documents(list(values))
        
        # Get the embedding for the search_value
        search_value_embedding = embeddings.embed_query(search_value)
        
        # Calculate cosine similarity between search_value and values
        similarities = []
        for value, value_embedding in zip(values, value_embeddings):
            similarity = cosine_similarity([search_value_embedding], [value_embedding])[0][0]
            similarities.append((value, similarity))
        
        # Sort values by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Pick the top 5 most similar values
        top_k_values = [value for value, _ in similarities[:5]]
        
        # Generate result string
        rtn_str = f"The columns below may contain relevant values for '{search_value}':"
        for value in top_k_values:
            for col in value_map[value]:
                rtn_str += f"\n- {col} contains '{value}'"
        
        return rtn_str
    
    def optimized_fuzzy_search(self,df, search_value, threshold=70, top_k=5, workers=4):
        """
        优化的全表模糊搜索
        """
        time=datetime.now()
        # 参数处理
        search_str = str(search_value).lower()
        search_columns = df.columns.tolist()
        
        # 预处理数据
        df_search = df[search_columns].copy()
        df_search = df_search.astype(str).apply(lambda x: x.str.lower())
        
        results = []
        
        def process_column(col_data, col_name):
            col_results = []
            for idx, val in col_data.items():
                if val == 'nan':
                    continue
                score = fuzz.token_set_ratio(search_str, val)
                if score >= threshold:
                    col_results.append({
                        'row_index': idx,
                        'column': col_name,
                        'value': df.at[idx, col_name],  # 原始值
                        'score': score
                    })
            return col_results
        
        # 并行处理各列
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_column, df_search[col], col): col for col in search_columns}
            for future in futures:
                results.extend(future.result())
        
        # 返回结果
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('score', ascending=False).head(top_k)

            rtn_str = f"The columns below may contain relevant values for '{search_value}':\n"
            # 正确遍历DataFrame的每一行
            for _, row in result_df.iterrows():
                rtn_str += f"\n- Column [{row['column']}] contains: '{row['value']}'"
            
            end_time = datetime.now()
            time_diff = end_time - time
            print(f"Optimized fuzzy search took: {time_diff.total_seconds()} seconds")

            return rtn_str
        
        else:
            end_time = datetime.now()
            time_diff = end_time - time
            print(f"Optimized fuzzy search took: {time_diff.total_seconds()} seconds")
            return f"No matches found for '{search_value}' in the table."
    
    def get_distinct_values(self, df, column_name,question):
        """
        获取 Pandas DataFrame 中列的唯一值。
        
        参数：
        - df: pandas DataFrame
        - column_name: 列名
        
        返回：
        - 列的唯一值列表
        """
        # sql_query = f"SELECT DISTINCT `{column}` FROM '{table}' LIMIT 50"
        column = df[column_name]
        values = column.dropna().astype(str).unique()
        if len(values) > 50:
            retriever = SentenceTransformerRetrieval(values, 'f:\TE\\all-mpnet-base-v2')
            values = retriever.get_top_k_sentences(question, k=50, distinct=True)
        rtn_str = f"The distinct values for {column} are: [{', '.join(values)}]. At most 50 values are shown due to length limit. Other values are omitted."
        return rtn_str
        
        # 如果有总计，平均等值，需要额外将这些值提取出来
        # 如果有格式明显与其他不同的值，也需要提取出来
        # return distinct_values[:10]  # 只返回前10个值
    


    def get_table_description(self, df, thought,question):
         # 随机选择 DataFrame 的 n 行数据
        # preview_data = df.sample(n=5).to_string(index=False)
        # 获取表格前五行的字符串表示
        preview_data = df.head(5).to_string(index=False)+ "\n" + df.tail(1).to_string(index=False)
        
        # 构建提示语句，向模型提供表格信息
        # prompt = f'''Here is a preview of a table's first few rows and last rows. 
        # Please generate a brief description of the table based on the question, including the column names and basic information for each column (such as data types, sample values, etc.) and relationship of rows and highlight the column values if related to the question.
        # \n\nTable preview:\n{preview_data}\n\nPlease describe the table. If there are any values like 'total' that are different from the other columns, please highlight them.'''
        # response = self.query(prompt)
        # print(response)
        prompt=f'''
Given a partial table preview along with a thought, focusing only on information relevant to the question and the thought, analyze the table and extract only the most relevant information.
Please Note:
- the given rows only contain a small portion of the table, not indicate the entire table.
- Directly output the most relevant columns that might be related to the question and the thought, and the column meaning of a column.
Output Format:
Directly output the columns and column meaning without any additional text or explanations.
Column name: Column Meaning
eg:
Column1: Column Meaning1
Column2: Column Meaning2
Table preview:\n{preview_data}\nQuestion:{question} Thought:{thought}.
        '''
        response = self.query(prompt)
        a=""
        for col in df.columns:
            # print(col)
            if col in response:
                # 获取该列的数据
                column = df[col]
                # 获取该列的数据类型
                column_dtype = column.dtype
                # 获取该列的前几个不同的值（根据实际情况你可以选择获取更多）
                unique_values = column.unique()

                if column_dtype == 'object':
                    text_values = [value for value in unique_values if (value is None or (isinstance(value, str) and not value.isdigit()))]
                    numeric_values = [value for value in unique_values if (value is not None and str(value).isdigit())]
                    # 如果包含文字的值不足10个，再从纯数字的值中补充
                    if len(text_values) < 7: 
                        sampled_values = text_values + np.random.choice(numeric_values, size=min(10 - len(text_values), len(numeric_values)), replace=False).tolist()
                    
                    
                    else:
                        text_values= np.random.choice(text_values, size=min(8, len(text_values)), replace=False)
                        sampled_values = np.random.choice(text_values, size=min(10, len(text_values)), replace=False)
                else:
                    # 如果数据类型不是object，直接取前10个值
                    sampled_values = np.random.choice(unique_values, size=min(10, len(unique_values)), replace=False)

                
                if unique_values.size < 10:
                    response += f"\nThe column '{col}' only has {len(sampled_values)} unique values: {unique_values}, and the column type is {column_dtype}."
                else :
                    response += f"\nThe column '{col}' has sampled values: {sampled_values}, and the column type is {column_dtype}."
        

            # if df[col].dtype == 'object':
            # # 找到包含关键词的行索引和值
            #     keywords = ['total', 'Total', 'Totaal']
            #     for keyword in keywords:
            #         mask = df[col].str.contains(keyword, case=False, na=False, regex=True)
            #         if mask.any():
            #             matched_rows = df.loc[mask, col]  # 获取匹配行的实际值
            #             total_index = matched_rows.index.tolist()
            #             matched_values = matched_rows.unique().tolist()  # 获取不重复的匹配值
                        
            #             # 构建包含实际值的消息
            #             a += f"""\nThe column '{col}' contains values matching '{matched_values}' at index {total_index}. 
            #             When performing calculations, these row(s) (identified by index {total_index}) should be excluded from the calculation. 
            #             Be cautious these row(s) are not involved in the calculation.\n"""

        return response+a
    



    def find_column_format_o(self, df, column_name,thought):
        """
        查找 Pandas DataFrame 中列的格式。
        类似于get_distinct_values, 查找数据可能的存在形式
        
        参数：
        - df: pandas DataFrame
        - column_name: 列名
        
        返回：
        - 生成的 Python 代码，用于将该列转化为统一的格式
        """
        # print(df.columns.tolist())
        # 获取该列的数据
        try:
            column = df[column_name]
        except KeyError:
            matching_columns = [col for col in df.columns if column_name in str(col)]
            if matching_columns:
                return f"Column '{column_name}' not found in the DataFrame. Did you mean one of these columns that contain '{column_name}': {', '.join(matching_columns)}?"
            else:   
                return f"Column '{column_name}' not found in the DataFrame."
        
        # 获取该列的数据类型
        column_dtype = column.dtype
        print(column)
        # 获取该列的前几个不同的值（根据实际情况你可以选择获取更多）
        unique_values = column.unique()
        # 筛选出包含文字的值
        if column_dtype == 'object':
            text_values = [value for value in unique_values if (value is None or (isinstance(value, str) and not value.isdigit()))]

            # 如果包含文字的值不足10个，再从纯数字的值中补充
            if len(text_values) < 18:
                numeric_values = [value for value in unique_values if (value is not None and value.isdigit())]
                sampled_values = text_values + np.random.choice(numeric_values, size=min(20 - len(text_values), len(numeric_values)), replace=False).tolist()
            else:
                sampled_values = np.random.choice(text_values, size=min(20, len(text_values)), replace=False)

            print(sampled_values)
            # sampled_values = np.random.choice(unique_values, size=min(10, len(unique_values)), replace=False)
            total_variants = [value for value in unique_values if isinstance(value, str) and re.search(r'total', value, re.IGNORECASE)]
    
            if total_variants:
            # 找到total的index
                total_index,name=self.get_total_index(df,column_name)
                response= f"""the column '{column_name}' has sampled values: {sampled_values}, and the column type is {column_dtype}."""
                response+= f"""\nThe column '{column_name}' contains the value '{name}' at index {total_index}, which is different from other rows. When performing calculations on this column, the row containing the value '{name}' (identified by index {total_index}) should be excluded from the calculation. Be cautious when processing this column to ensure that the 'total' row is not involved in the calculation."""
                return response
        else:
            # 如果数据类型不是object，直接取前10个值
            sampled_values = np.random.choice(unique_values, size=min(20, len(unique_values)), replace=False)

        # 构建查询提示，询问 LLM 生成 Python 代码处理该列格式问题
        prompt = f"""
The column type of '{column_name}' is {column_dtype}.
Here are some example values from the column: {sampled_values}.
You should determine whether there is a need to process the column using Python code for the given purpose. 
Determine whether the given column needs processing using Python based on its content and format.
The column can be processed by df["{column_name}"]
Purpose: {thought}
Notes:
If the column's format needs additional transformation to calculate the result based on the purpose, then it needs to be processed and transformed into the correct type.
For example:
- If sampled_values are ['1,200,300', '1,300,299'], they need to be processed to ['1200300', '1300299'].
- Use df to process the column, and output the code in the format of:
```python
# your code here
```
- Do not output any additional text or create any sample data.
- Do not sample DataFrame or generate any additional data. Only generate code to process the column.
- The sampled values are not the entire table, just a few examples of the column's data.
- Iust process the format of the column, do not do o=additional operations, such as calculating the mean or sum.
- If the column's format is already correct, just return "[[No]]".
"""
        # 使用 LLM 来处理和生成代码
        response= self.query(prompt)  # 假设 `llm_client.query` 方法可以发送查询并获得响应
        # 用正则表达式提取 ```python\n``` 之间的代码
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else None
        # print(prompt)
        print(response)
        while extracted_code:
            # print(extracted_code)  # 输出提取的代码
            try:
                exec(extracted_code)
                unique_values = df[column_name].unique()[:10]
                response+=f"\n\nThe column '{column_name}' has been processed, here are are example values form the column: {unique_values}."
            except Exception as e:
                extracted_code = self.critic.critic_python_tool(extracted_code,e,column_name)
                # response+=f"\n\nAn error occurred while processing the column '{column_name}': {e}"
        # print(df)
        else:
            response=f"the column '{column_name}' has sampled values: {sampled_values}, and the column type is {column_dtype}.\n\n"
        # 返回 LLM 给出的 Python 代码（用于统一列格式）
        return response

    def find_column_format(self, df, column_name,thought,crticapi,iter_count=0):
        """
        查找 Pandas DataFrame 中列的格式。
        类似于get_distinct_values, 查找数据可能的存在形式
        
        参数：
        - df: pandas DataFrame
        - column_name: 列名
        
        返回：
        - 生成的 Python 代码，用于将该列转化为统一的格式
        """
        # print(df.columns.tolist())
        # 获取该列的数据
        try:
            column = df[column_name]
        except KeyError:
            matching_columns = [col for col in df.columns if column_name in str(col)]
            if matching_columns:
                return f"Column '{column_name}' not found in the DataFrame. There is a column that you may want to find [{matching_columns}]",0
            else:   
                return f"Column '{column_name}' not found in the DataFrame.",0
        
        # 获取该列的数据类型
        column_dtype = column.dtype
        print(column)
        # 获取该列的前几个不同的值（根据实际情况你可以选择获取更多）
        unique_values = column.unique()
        # 筛选出包含文字的值
        if column_dtype == 'object':
            text_values = [value for value in unique_values if (value is None or (isinstance(value, str) and not value.isdigit()))]

            # 如果包含文字的值不足10个，再从纯数字的值中补充
            if len(text_values) < 18:
                numeric_values = [value for value in unique_values if (value is not None and value.isdigit())]
                sampled_values = text_values + np.random.choice(numeric_values, size=min(20 - len(text_values), len(numeric_values)), replace=False).tolist()
            else:
                sampled_values = np.random.choice(text_values, size=min(20, len(text_values)), replace=False)

            print(sampled_values)
            # sampled_values = np.random.choice(unique_values, size=min(10, len(unique_values)), replace=False)
            # total_variants = [value for value in unique_values if isinstance(value, str) and re.search(r'total', value, re.IGNORECASE)]
    
            # if total_variants:
            # # 找到total的index
            #     total_index,name=self.get_total_index(df,column_name)
            #     # response= f"""the column '{column_name}' has sampled values: {sampled_values}, and the column type is {column_dtype}."""
            #     response+= f"""\nThe column '{column_name}' contains the value '{name}' at index {total_index}, which is different from other rows. When performing calculations on this column, the row containing the value '{name}' (identified by index {total_index}) should be excluded from the calculation. Be cautious when processing this column to ensure that the 'total' row is not involved in the calculation."""
                # return response
        else:
            # 如果数据类型不是object，直接取前10个值
            sampled_values = np.random.choice(unique_values, size=min(20, len(unique_values)), replace=False)

        # 构建查询提示，询问 LLM 生成 Python 代码处理该列格式问题
        prompt = f"""
The column type of '{column_name}' is {column_dtype}.
Here are some example values from the column: {sampled_values}.
You should determine whether there is a need to process the column using Python code for the given purpose. 
Determine whether the given column needs processing using Python based on its content and format.
The column can be processed by df["{column_name}"]
Purpose: {thought}
Notes:
If the column's format needs additional transformation to calculate the result based on the purpose, then it needs to be processed and transformed into the correct type.
For example:
- If sampled_values are ['1,200,300', '1,300,299'], they need to be processed to ['1200300', '1300299'].
- Use df to process the column, and output the code in the format of:
```python
# your code here
```
- Do not output any additional text or create any sample data.
- Do not sample DataFrame or generate any additional data. Only generate code to process the column.
- The sampled values are not the entire table, just a few examples of the column's data.
- Just process the format of the column, do not do additional operations, such as calculating the mean or sum.
- If the column's format is already correct, just return "[[No]]".
"""
        # 使用 LLM 来处理和生成代码
        text= self.query(prompt)  # 假设 `llm_client.query` 方法可以发送查询并获得响应
        # 用正则表达式提取 ```python\n``` 之间的代码
        if "[[No]]" in text:
            response= f"the column '{column_name}' has sampled values: {sampled_values}, and the column type is {column_dtype}.\n\n"
            return response,iter_count
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else None
        # print(prompt)
        response=""
        while 1:
            # print(extracted_code)  # 输出提取的代码
            try:
                exec(extracted_code)
                unique_values = df[column_name].unique()[:5]
                response+=f"\n\nThe column '{column_name}' has been processed, here are are example values form the column: {unique_values}."
                break
            except Exception as e:
                print(f"Error executing code: {e}")
                if iter_count>3:
                    response+=f"\n\nWe cannot process the format of the column '{column_name}' , and the column type is {column_dtype}.\n\n"
                    break
                iter_count+=1
                extracted_code = crticapi.critic_python_tool(extracted_code,e,sampled_values,thought,column_dtype)
                # response+=f"\n\nAn error occurred while processing the column '{column_name}': {e}"
                # response+=f"\n\nAn error occurred while processing the column '{column_name}': {e}"
       
        print(extracted_code)  # 输出提取的代码
        # try:
        #     exec(extracted_code)
        #     unique_values = df[column_name].unique()[:5]
        #     response+=f"\n\nThe column '{column_name}' has been processed, here are are example values form the column: {unique_values}."
        # except Exception as e:
        #     print(f"Error executing code: {e}")
        #     response+=f"\n\nAn error occurred while processing the column '{column_name}': {e}"
        # print(df)
        return response,iter_count


    def get_total_index(self,df,column_name):
        """
        获取列中值为 'total' 的行索引。
        
        参数：
        - df: pandas DataFrame
        - column_name: 列名
        
        返回：
        - 行索引列表
        """
        # 获取该列的数据
        column = df[column_name]
        # 查找值为 'total' 的行索引
        total_index = column.index[column.str.contains((r'total|totaal'), case=False, na=False, regex=True)].tolist()
        name=column[total_index]
        return total_index,name

    def process_column_format(self,df,column_name):
        """
        统一对列进行处理，将其转化为指定的格式
        需要调用api或者是写代码对列进行转换
        
        参数：
        - df: pandas DataFrame
        - column: 列名
        - format: 要转化的格式
        
        返回：
        - 格式化后的 DataFrame
        
        """
        prompt=f"列'{column_name}' 存在什么格式，是否需要转化成统一的格式"
        # get top-k similarity values
        # query the LLM to process the table，比如说格式不统一或者日期个人格式不同
        column = df[column_name]
        # 获取该列可能的值，把值转化为需要的格式，比如将分数转化为输赢
        print(f"列 '{column_name}' 的格式为：{column.dtype}")        
        return df
    

def extract_action(response):
    if "df" in response.lower() or "python " in response.lower():
        action=response.split("Action:")[-1].strip()
        return [action]
    # 使用正则表达式提取所有 Action
    actions = re.findall(r'Action:\s*(.*)', response)
    
    # 如果没有找到任何 Action
    if not actions:
        return "No actions detected."
    
    # 提取工具名称和参数
    tool_info = []
    for action in actions:
        # 如果 action 中包含逗号，说明可能存在多个工具调用
        if ',' in action:
            # 分割逗号，提取每个工具调用
            sub_actions = re.split(r',\s*', action)
            for sub_action in sub_actions:
                # 匹配工具名称和参数
                match = re.match(r'(\w+)\((.*)\)', sub_action)
                if match:
                    tool_name = match.group(1)
                    tool_value = match.group(2)
                else:
                    # 如果没有参数
                    tool_name = sub_action.strip()
                    tool_value = None
                tool_info.append((tool_name, tool_value))
        else:
            # 匹配工具名称和参数
            match = re.match(r'(\w+)\((.*)\)', action)
            if match:
                tool_name = match.group(1)
                tool_value = match.group(2)
            else:
                # 如果没有参数
                tool_name = action.strip()
                tool_value = None
            tool_info.append((tool_name, tool_value))
    
    # 检查工具名称是否相同
    tool_names = [info[0] for info in tool_info]
    if len(set(tool_names)) == 1:
        # 如果所有工具名称相同
        tool_values = [f"{tool_name}({tool_value})" if tool_value else tool_name for tool_name, tool_value in tool_info]
        # print(f"Extracted tool values: {tool_values}")
        return  tool_values
    else:
        # 如果工具名称不同
        return  False
    
        