import sys
import time
import json
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMGRAG_ROOT = REPO_ROOT / 'SimGRAG'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIMGRAG_ROOT))
sys.path.insert(0, str(SIMGRAG_ROOT / 'src'))

from src.llm import LLM
import prompts.answer_ship
import prompts.rewrite_ship
from src.dataset import MetaQA
from src.retriever import Retriever
from src.utils import check_answer
from src.utils import extract_graph
from src.dataset import ShipQA 

configs_path = SIMGRAG_ROOT / 'configs' / 'ShipQA.json'
configs = json.load(open(str(configs_path), 'r', encoding='utf-8'))

def _resolve_cfg_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((SIMGRAG_ROOT / p).resolve())

for key in ('raw_data_dir', 'processed_data_dir', 'output_filename'):
    if key in configs:
        try:
            configs[key] = _resolve_cfg_path(configs[key])
        except Exception:
            pass
try:
    Path(configs.get('processed_data_dir', SIMGRAG_ROOT / 'data' / 'ShipQA')).mkdir(parents=True, exist_ok=True)
    out_path = Path(configs.get('output_filename', SIMGRAG_ROOT / 'results' / 'ship_query_results.txt'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

dataset = ShipQA(configs)
KG = dataset.get_KG()
all_queries = dataset.get_queries()
all_groundtruths = dataset.get_groundtruths()

llm = LLM(configs)

retriever = Retriever(configs, KG)

def run(query, groundtruths):
    res = {
        'query': query,
        'groundtruths': groundtruths,
        'retriever_configs': configs['retriever'],
        'llm_configs': configs['llm'],
        'rewrite_shot': configs['rewrite_shot'],
        'answer_shot': configs['answer_shot'],
    }
    
    try:
        start = time.time()
        res['rewrite_prompt'] = prompts.rewrite_ship.get(query, shot=res['rewrite_shot'])
        res['rewrite_llm_output'] = llm.chat(res['rewrite_prompt'])
        res['rewrite_time'] = time.time() - start
  
        res['query_graph'] = extract_graph(res['rewrite_llm_output'])
        
        start = time.time()
        res['retrieval_details'] = retriever.retrieve(res['query_graph'], mode='greedy')
        res['evidences'] = [each[1] for each in res['retrieval_details']['results']]
        res['retrieval_time'] = time.time() - start

        start = time.time()
        res['answer_prompt'] = prompts.answer_ship.get(res['query'], res['evidences'], shot=res['answer_shot'])
        res['answer_llm_output'] = llm.chat(res['answer_prompt'])
        res['answer_time'] = time.time() - start
  
        res['correct'] = check_answer(res['answer_llm_output'], groundtruths)
    
    except Exception as e:
        res['error_message'] = str(e)
        res['correct'] = False
  
    return res
	
result_file = configs["output_filename"]

start_index = 11 

combined_data = list(zip(all_queries, all_groundtruths))


for query, groundtruths in tqdm(combined_data[start_index:], total=len(combined_data) - start_index):
    res = run(query, groundtruths)
    
    with open(result_file, 'a', encoding='utf-8') as f:
        try:
            line = json.dumps(res, ensure_ascii=False)
            f.write(line + '\n')
        except TypeError as e:
            print(f"\n JSON 写入失败，错误查询：{query}")
            f.write(json.dumps({"error": "Serializing Error", "query": query}, ensure_ascii=False) + '\n')

print(f"结果已追加至: {result_file}")