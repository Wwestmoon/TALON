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
print(sys.executable)
import os
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


import pandas as pd
from tqdm import tqdm

# from agent import TableAgent, TableRAGAgent

from agent import TableAgentSC
from evaluate import evaluate
from utils.load_data import load_dataset

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = f""

def solve(args):
    agent_args, data, sc_id = args
    if agent_args['agent_type'] in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling', 'TableTools']:
        agent = TableAgentSC(**agent_args)
    else:
        raise NotImplementedError(f"Agent type {agent_args['agent_type']} not supported.")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return agent.run(data, sc_id=sc_id)


def main(

    dataset_path = r'',

    model_name = 'gpt-4o-mini',
    agent_type = 'TableTools',
    retrieve_mode = 'embed',
    embed_model_name = 'text-embedding-3-large',

    log_dir = '',
    db_dir = 'db/',
    top_k = 5,
    sr = 0, # self-refine, deprecated
    sc =5, # self-consistency
    max_encode_cell = 10000,
    stop_at =500,
    match="fuzzy",
    resume_from =0,
    load_exist = True, 
    n_worker = 1,
    verbose = True,
):
    os.makedirs(os.path.join(log_dir, 'log'), exist_ok=True)

    # store the config
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird'] if task_name in dataset_path][0]
    db_dir = os.path.join(db_dir, task + '_' + Path(dataset_path).stem)
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump({key: value for key, value in locals().items() if key != 'fp'}, fp, indent=4)

    dataset = load_dataset(task, dataset_path, stop_at)

    
    if stop_at < 0:
        stop_at = len(dataset)

    agent_args = {
        'model_name': model_name,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': agent_type,
        'top_k': top_k,
        'sr': sr,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'load_exist': load_exist,
        'verbose': verbose,
        'match': match,
        'sc': sc,
    }

    results = []
    for data in tqdm(dataset[resume_from:stop_at]):

            result = solve((agent_args, data, 0))
            results.append(result)

    acc = evaluate(task, results)
    print(f'Accuracy: {acc}')
    # stats_keys = ['n_iter', 'init_prompt_token_count', 'total_token_count']
    stats_keys = ['n_iter', 'total_token_count']
    stats_df = pd.DataFrame.from_records(results)[stats_keys]
    print(stats_df.describe().to_string())

    # store the result
    result_dict = stats_df.mean().to_dict()
    result_dict['accuracy'] = acc
    for key in ['model_name', 'retrieve_mode', 'embed_model_name', 'task', 'agent_type', 'top_k', 'max_encode_cell', 'sr']:
        result_dict[key] = agent_args[key]
    result_dict['sc'] = sc
    result_dict['data'] = Path(dataset_path).stem
    result_path = os.path.join(log_dir, 'result.json')
    with open(result_path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)





if __name__ == '__main__':
    # fire.Fire(main)
    main()


