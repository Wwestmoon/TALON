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

import json
from tqdm import tqdm
from pathlib import Path
import random
seed=42
def load_dataset(task, dataset_path, stop_at=-1):
    dataset = []
    tag = Path(dataset_path).stem
    # with open(dataset_path, encoding="utf-8") as fp:
    #     all_lines = fp.readlines()
    # all_lines = all_lines[:stop_at]
    # load json file
    random.seed(seed)
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = json.load(f)
    print(len(all_lines))
    random.shuffle(all_lines)
    all_lines = random.sample(all_lines, min(stop_at, len(all_lines)))

    
    for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc=f"Loading {task}-{tag} dataset"):
        # info = json.loads(line)
        info = line
        if 'id' in info:
            info['orig_id'] = info['id']
        info["id"] = f"{tag}-{i}"
        if 'table_id' not in info:
            info['table_id'] = info['table_caption'].replace(' ', '_')
        dataset.append(info)

        random.seed(42)

    return dataset