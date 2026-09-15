#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import traceback
import re
import logging
import builtins

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
_orig_print = builtins.print
def _print_to_stderr_by_default(*args, **kwargs):
    if 'file' in kwargs:
        return _orig_print(*args, **kwargs)
    return _orig_print(*args, **{**kwargs, 'file': sys.stderr})
builtins.print = _print_to_stderr_by_default

from init_rags import init_lightrag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', required=True)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'SemanticAligner' / 'configs' / 'ShipQA.json'))
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    SemanticAligner_root = repo_root / 'SemanticAligner'
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(SemanticAligner_root))
    sys.path.insert(0, str(SemanticAligner_root / 'src'))

    def _print_error_and_exit(exc: Exception):
        err = {'error': str(exc)}
        if os.environ.get('DEBUG_QUERY_SINGLE') == '1':
            tb = traceback.format_exc()
            err['traceback_lines'] = tb.splitlines()
            print('ERROR in query_single:', file=sys.stderr)
            traceback.print_exc()
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
        return

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from src.vecdb import VectorStore
    except Exception as e:
        return _print_error_and_exit(e)

    try:
        configs = json.load(open(args.config, 'r', encoding='utf-8'))
    except Exception as e:
        return _print_error_and_exit(e)

    try:
        SemanticAligner_ROOT = SemanticAligner_root
        def _resolve_cfg_path(p: str):
            if not isinstance(p, str):
                return p
            path = Path(p)
            if path.is_absolute():
                return str(path)
            return str((SemanticAligner_ROOT / p).resolve())

        for key in ('raw_data_dir', 'processed_data_dir', 'output_filename'):
            if key in configs:
                try:
                    configs[key] = _resolve_cfg_path(configs[key])
                except Exception:
                    pass

        try:
            Path(configs.get('processed_data_dir', SemanticAligner_ROOT / 'data' / 'ShipQA')).mkdir(parents=True, exist_ok=True)
            out_path = Path(configs.get('output_filename', SemanticAligner_ROOT / 'results' / 'ship_query_results.txt'))
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    except Exception:
        pass

    try:
        from src.llm import LLM
        import prompts.rewrite_ship as rewrite_ship
        from src.utils import extract_graph
        from src.dataset import ShipQA
        from src.retriever import Retriever
    except Exception as e:
        return _print_error_and_exit(e)

    try:
        dataset = ShipQA(configs)
        KG = dataset.get_KG()

        llm = LLM(configs)
        retriever = Retriever(configs, KG)

        rewrite_prompt = rewrite_ship.get(args.question, shot=configs.get('rewrite_shot', 0))
        rewrite_llm_output = llm.chat(rewrite_prompt)

        query_graph = extract_graph(rewrite_llm_output)
        def _sanitize_graph(g):
            out = []
            if not isinstance(g, list):
                return out
            for item in g:
                if isinstance(item, dict):
                    keys = list(item.keys())
                    if len(keys) >= 3:
                        out.append([str(item[keys[0]]), str(item[keys[1]]), str(item[keys[2]])])
                    else:
                        vals = list(item.values())
                        if len(vals) >= 3:
                            out.append([str(vals[0]), str(vals[1]), str(vals[2])])
                    continue
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3:
                        out.append([str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()])
                    else:
                        continue
                elif isinstance(item, str):
                    m = None
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list) and len(parsed) >= 3:
                            out.append([str(parsed[0]), str(parsed[1]), str(parsed[2])])
                            continue
                    except Exception:
                        pass
                    parts = [p.strip().strip('"\'') for p in re.split(r'[,，|]', item) if p.strip()]
                    if len(parts) >= 3:
                        out.append([parts[0], parts[1], parts[2]])
            return out

        sanitized_qg = _sanitize_graph(query_graph)
        if not sanitized_qg:
            raise Exception(f"Invalid or empty query_graph after rewrite. rewrite output snippet: {str(rewrite_llm_output)[:300]}")

        retrieval_details = retriever.retrieve(sanitized_qg, mode='greedy')
        evidences = [each[1] for each in retrieval_details.get('results', [])]

        res = {
            'query': args.question,
            'retrieval_details': retrieval_details,
            'evidences': evidences,
            'rewrite_llm_output': rewrite_llm_output,
        }

    except Exception as e:
        return _print_error_and_exit(e)

    candidates = []
    rd = res.get('retrieval_details') or {}
    results = rd.get('results', []) if isinstance(rd, dict) else []

    for item in results:
        try:
            score, graph, reuse_nodes = item
        except Exception:
            continue
        try:
            if isinstance(graph, list) and len(graph) > 0 and isinstance(graph[0], (list, tuple)):
                rep_text = str(graph[0][0]).strip()
            else:
                parts = []
                for h, r, t in graph:
                    parts.append(f"{h} {r} {t}")
                rep_text = '；'.join(parts)
        except Exception:
            rep_text = str(graph)
        try:
            sc = float(score)
        except Exception:
            sc = 1.0
        candidates.append({'text': rep_text, 'score': sc, 'source': 'SemanticAligner'})

    if not candidates and 'evidences' in res:
        for ev in res.get('evidences', []):
            try:
                if isinstance(ev, list) and len(ev) > 0 and isinstance(ev[0], (list, tuple)):
                    rep_text = str(ev[0][0]).strip()
                else:
                    parts = [f"{h} {r} {t}" for h, r, t in ev]
                    rep_text = '；'.join(parts)
            except Exception:
                rep_text = str(ev)
            candidates.append({'text': rep_text, 'score': 1.0, 'source': 'SemanticAligner'})

    seen_texts = set()
    unique_candidates = []
    for c in candidates:
        txt = c.get('text', '').strip()
        if txt and txt not in seen_texts:
            seen_texts.add(txt)
            unique_candidates.append({'text': txt, 'score': c.get('score', 1.0), 'source': c.get('source', 'SemanticAligner')})

    print(json.dumps({'candidates': unique_candidates}, ensure_ascii=False), file=sys.stdout)


if __name__ == '__main__':
    main()
