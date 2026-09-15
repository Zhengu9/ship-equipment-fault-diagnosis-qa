#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple


_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    return _orig_print(*args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parent
HF_ENDPOINT_DEFAULT = 'https://hf-mirror.com'
_hf_raw = os.environ.get('HF_ENDPOINT')
if _hf_raw:

    _hf_s = _hf_raw.replace('、', '').strip()
    if '/api/models' in _hf_s:
        _hf_s = _hf_s.split('/api/models', 1)[0]

    _hf_s = _hf_s.rstrip('/')
    os.environ['HF_ENDPOINT'] = _hf_s
    try:
        print(f"[init_rags] HF_ENDPOINT set to {_hf_s} (from env)")
    except Exception:
        pass
else:
    os.environ['HF_ENDPOINT'] = HF_ENDPOINT_DEFAULT
    try:
        print(f"[init_rags] HF_ENDPOINT is ready")
    except Exception:
        pass

SENTINEL_PATH = REPO_ROOT / '.lightrag_initialized'

RAG_INSTANCES = {}

async def _process_dataset(p: Path, embed_model: str = 'BAAI/bge-m3', llm: str = 'qwen:7b'):

    from sentence_transformers import SentenceTransformer
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc

    try:
        st = SentenceTransformer(embed_model, trust_remote_code=True)
    except Exception:
        st = None

    async def local_embed_func(texts):
        if isinstance(texts, str):
            texts = [texts]

        res = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return res

    async def llm_model_func(prompt, **kwargs):
        from subprocess import run
        proc = run(['ollama', 'run', llm], input=prompt, text=True, capture_output=True)
        return proc.stdout.strip() or proc.stderr.strip()

    embedding_dim = 1024 if st is not None and hasattr(st, 'get_sentence_embedding_dimension') and st.get_sentence_embedding_dimension() else 768
    embedding_func = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=local_embed_func)

    config = RAGAnythingConfig(working_dir=str(p), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
    rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)

    init_file = REPO_ROOT / 'FullDocRetriever' / 'init.txt'
    if not init_file.exists():
        init_file.write_text('init', encoding='utf-8')

    out_dir = p / 'output'
    out_dir.mkdir(exist_ok=True)

    await rag.process_document_complete(file_path=str(init_file), output_dir=str(out_dir), parse_method='auto')

    return rag


def init_lightrag(dataset_dirs: List[Tuple[str, Path]], embed_model: str = 'BAAI/bge-m3', llm: str = 'qwen:7b'):

    if not dataset_dirs:
        print('[init_rags] no dataset dirs provided; nothing to do')
        return

    processed_meta = []
    for name, p in dataset_dirs:
        key = name or str(p)
        try:
            status_file = p / 'kv_store_doc_status.json'
            already_processed = False
            if status_file.exists():
                try:
                    sf = json.load(open(status_file, 'r', encoding='utf-8'))
                    for v in sf.values():
                        if isinstance(v, dict) and v.get('status') == 'processed':
                            already_processed = True
                            break
                except Exception:
                    already_processed = False

            if already_processed:
                RAG_INSTANCES[key] = {'path': str(p), 'ready': True}
                processed_meta.append((name, p, None))
                print(f"[init_rags] skipping {p}, already processed")
                continue

            try:
                rag = asyncio.run(_process_dataset(p, embed_model=embed_model, llm=llm))
            except Exception as eproc:
                print(f"[init_rags] failed processing {p}: {eproc}", file=sys.stderr)
                raise

            RAG_INSTANCES[key] = {'path': str(p), 'ready': True}
            processed_meta.append((name, p, rag))
        except Exception as e:
            print(f"[init_rags] failed processing {p}: {e}", file=sys.stderr)
            raise

    try:
        meta = {'time': time.time(), 'datasets': [str(p.relative_to(REPO_ROOT)) if p.is_absolute() else str(p) for (_, p, _) in processed_meta]}
        SENTINEL_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        print(f"[init_rags] wrote sentinel {SENTINEL_PATH}")
    except Exception as e:
        print(f"[init_rags] warning: failed to write sentinel: {e}")


def is_initialized() -> bool:
    return SENTINEL_PATH.exists()


def run_cmd(cmd, cwd: Optional[str] = None, env: Optional[dict] = None, timeout: Optional[int] = None):
    res = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        res.check_returncode()
    return res


def init_GraphReasoner(root: str):
    root = Path(root)
    GraphReasoner_dir = root / 'GraphReasoner'
    if not GraphReasoner_dir.exists():
        print(f"GraphReasoner not found at {GraphReasoner_dir}, skip.")
        return
    dataset_dir = GraphReasoner_dir / 'data' / 'ship_cwq'
    if dataset_dir.exists():
        print(f"GraphReasoner dataset found at {dataset_dir}; skipping extraction/prep as requested.")
    else:
        print(f"Warning: expected GraphReasoner dataset not found at {dataset_dir}. Please verify data placement.")
    marker = GraphReasoner_dir / '.init_done'
    if not marker.exists():
        try:
            marker.write_text(f'initialized; dataset={dataset_dir}')
            print('Created init marker for GraphReasoner (no files modified).')
        except Exception as e:
            print(f'Failed to write GraphReasoner init marker: {e}')
    else:
        print('GraphReasoner already marked initialized; nothing changed.')


def init_SemanticAligner(root: str):
    root = Path(root)
    SemanticAligner_dir = root / 'SemanticAligner'
    if not SemanticAligner_dir.exists():
        print(f"SemanticAligner not found at {SemanticAligner_dir}, skip.")
        return
    pipeline_dir = SemanticAligner_dir / 'pipeline'
    if pipeline_dir.exists():
        print(f"SemanticAligner pipeline found at {pipeline_dir}; skipping indexing step as requested.")
    else:
        print(f"Warning: SemanticAligner pipeline directory not found at {pipeline_dir}.")
    marker = SemanticAligner_dir / '.init_done'
    if not marker.exists():
        try:
            marker.write_text('initialized (no indexing performed)')
            print('Created init marker for SemanticAligner (no files modified).')
        except Exception as e:
            print(f'Failed to write SemanticAligner init marker: {e}')
    else:
        print('SemanticAligner already marked initialized; nothing changed.')


def wait_for_milvus(host: str = '127.0.0.1', port: int = 19530, timeout: int = 300, interval: float = 2.0):
    import socket
    deadline = time.time() + float(timeout)
    print(f"Waiting for Milvus at {host}:{port} (timeout={timeout}s)")
    while True:
        try:
            with socket.create_connection((host, port), timeout=3):
                print('Milvus is reachable')
                return True
        except Exception:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for Milvus at {host}:{port}")
            time.sleep(interval)


def ensure_ollama_model(model: str = 'qwen:7b', pull: bool = True, server_script: Optional[str] = None, start_server: bool = False):
    if pull:
        try:
            res = subprocess.run(['ollama', 'list'], text=True, capture_output=True, check=True)
            stdout = (res.stdout or '')
            if model in stdout:
                print(f"Ollama model '{model}' already present, skipping pull.")
            else:
                print(f"Ollama model '{model}' not found locally, pulling...")
                run_cmd(['ollama', 'pull', model])
        except Exception as e:
            print('Failed to query ollama models, attempting to pull anyway:', e)
            try:
                run_cmd(['ollama', 'pull', model])
            except Exception as e2:
                print(f'ollama pull failed: {e2}')
    if start_server and server_script:
        server_path = Path(server_script)
        if server_path.exists():
            print(f'Starting ollama server via {server_path} (detached)')
            subprocess.Popen(['bash', str(server_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f'Ollama server script not found: {server_script}')


def start_milvus_compose(compose_path: Optional[Path] = None):

    if compose_path is None:
        candidate = REPO_ROOT / 'SemanticAligner' / 'pipeline' / 'docker-compose.yml'
        if candidate.exists():
            compose_path = candidate
        else:
            compose_path = REPO_ROOT / 'docker-compose.yml'
    else:
        compose_path = Path(compose_path)

    if not compose_path.exists():

        return

    target_dir = str(compose_path.parent)
    old_cwd = os.getcwd()
    last_err = None
    try:
        os.chdir(target_dir)
        try:
            cmd = ['docker-compose', 'up', '-d']
            run_cmd(cmd, cwd=target_dir)

            return
        except Exception as e:
            last_err = e

        try:
            cmd2 = ['docker', 'compose', '-f', str(compose_path), 'up', '-d']
            print(f"[init_rags] Fallback running: {' '.join(cmd2)} (cwd={target_dir})")
            run_cmd(cmd2, cwd=target_dir)
            print(f"[init_rags] docker compose fallback succeeded with: {' '.join(cmd2)}")
            return
        except Exception as e2:
            print(f"[init_rags] docker compose fallback failed: {e2}", file=sys.stderr)
            last_err = e2

    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass

    print(f"[init_rags] All compose attempts failed for {compose_path}; last error: {last_err}", file=sys.stderr)


def init_FullDocRetriever_stub(root: str):

    root = Path(root)
    any_dir = root / 'FullDocRetriever'
    if not any_dir.exists():
        print(f"FullDocRetriever not found at {any_dir}, skip.")
        return
    init_marker = any_dir / '.init_done'

    dataset_names = ['rag_storage', 'rag_storage_guanlunji', 'rag_storage_guide', 'rag_storage_maintain', 'rag_storage_qilunji', 'rag_storage_reportpaper', 'rag_storage_security']
    any_found = False
    for name in dataset_names:
        p = any_dir / name
        if not p.exists():
            continue
        any_found = True
        out_dir = p / 'output'
        if out_dir.exists():
            print(f"[FullDocRetriever] existing output for {name} found at {out_dir}; leaving unchanged.")
        else:
            print(f"[FullDocRetriever] dataset {name} exists but no output dir; skipping processing as requested.")

        try:
            init_lightrag([(name, p)])
        except Exception as e:
            print(f"[FullDocRetriever] init_lightrag failed for {p}: {e}", file=sys.stderr)
        marker = p / '.init_done'
        if not marker.exists():
            try:
                marker.write_text('initialized')
                print(f"[FullDocRetriever] wrote marker for dataset {name} (no processing).")
            except Exception as e:
                print(f"[FullDocRetriever] failed to write marker for {name}: {e}")
    try:
        init_marker.write_text('initialized')
    except Exception as e:
        print(f"[FullDocRetriever] failed to write top-level init marker: {e}")
    if any_found:
        print('[FullDocRetriever] initialization stub completed (no files modified).')
    else:
        print('[FullDocRetriever] no datasets detected; created top-level marker where possible.')


def init_all(root: str = '.'):
    root = str(Path(root).resolve())
    print(f"Initializing all RAG projects under: {root}")

    try:
        try:
            start_milvus_compose()
        except Exception:
            pass
        wait_for_milvus()
    except Exception as e:
        print(f"Warning: Milvus readiness check failed: {e}")

    try:
        ensure_ollama_model()
    except Exception as e:
        print(f"Warning: ensure_ollama_model failed: {e}")

    init_GraphReasoner(root)
    init_SemanticAligner(root)


def main(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description='Initialize RAG projects (FullDocRetriever, GraphReasoner, SemanticAligner)')
    parser.add_argument('--project', choices=['FullDocRetriever', 'GraphReasoner', 'SemanticAligner', 'All'], default='All')
    parser.add_argument('--root', default='.', help='repo root path')
    parser.add_argument('--embed-model', default='BAAI/bge-m3')
    parser.add_argument('--llm', default='qwen:7b')
    parser.add_argument('--start-ollama', action='store_true')
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()

    if args.project in ('FullDocRetriever', 'All'):
        any_base = Path(root) / 'FullDocRetriever'
        dataset_names = ['rag_storage', 'rag_storage_guanlunji', 'rag_storage_guide', 'rag_storage_maintain', 'rag_storage_qilunji', 'rag_storage_reportpaper', 'rag_storage_security']
        dataset_dirs = []
        for name in dataset_names:
            p = any_base / name
            if p.exists():
                dataset_dirs.append((name, p))
        if dataset_dirs:
            try:
                init_lightrag(dataset_dirs)
            except Exception as e:
                print(f"[init_rags] init_lightrag failed: {e}")
        else:
            init_FullDocRetriever_stub(str(root))

    if args.project in ('GraphReasoner', 'all'):
        init_GraphReasoner(str(root))

    if args.project in ('SemanticAligner', 'all'):
        init_SemanticAligner(str(root))


if __name__ == '__main__':
    main()
