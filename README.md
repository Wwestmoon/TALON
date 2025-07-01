# TALON: A Multi-Agent Framework for Long-Table Exploration and Question Answering


## Environment 

```shell
conda create --name talon python=3.10 -y
conda activate talon
pip install -r requirements.txt
```

## Data
Download datasets and pre-built databases from [here](https://drive.google.com/drive/folders/1tWa79UzwV-e9-vvyIp23d5uav3koyx-h?usp=drive_link).
- `data/wtq/wtq_l.json`: WTQ-L test set.
- `data/bird/bird_314.json`: BirdQA test set.

## Code Structure
- `run.py`: Main script to execute experiments.
- `utils/wtq_official_eval.py`: Evaluates the results stored in the output directory.
- `agent/agent_sc.py`: Implementation of the TALON.
- `agent/model.py`: Manages calls to LLM APIs from OpenAI and other Models
- `agent/tool.py`: Handles building databases and performs schema/cell/row/column retrieval.

## Usage

### Command Arguments

- `--dataset_path`: Path to the dataset, default: `data/wtq/wtq_l.json`
- `--model_name`: Name of the model, default: `gpt-4o-mini`, options: `text-bison@001`, `text-bison@002`, `text-unicorn@001`
- `--log_dir`: Directory for logs, default: 'output/test/'
- `--db_dir`: Directory for databases, default: 'db/'
- `--top_k`: Number of retrieval results, default: 5
- `--sc`: Self-consistency, default: 5
- `--stop_at`: Stopping point, default: -1 means no specific stop
- `--resume_from`: Point to start/resume from, default: 0
- `--load_exist`: Load existing results, default: False
- `--n_worker`: Number of workers, default: 1
- `--verbose`: Verbose output, default: False  

### Examples

Run and evaluate TableRAG on the ArcadeQA dataset:

```shell
python run.py \
--dataset_path data/wtq/wtq_l.json \
--model_name gpt-4o-mini \
--log_dir 'output/wtq' \
--top_k 5 \
--sc 5 \
--n_worker 16
```
