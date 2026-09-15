#!/usr/bin/env python3
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from init_rags import init_lightrag, RAG_Retriever, init_FullDocRetriever_stub


def run_ollama(model: str, prompt: str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(['ollama', 'run', model], input=prompt, text=True, capture_output=True, timeout=timeout)
        return proc.stdout.strip() or proc.stderr.strip()
    except Exception as e:
        return f"<ollama error: {e}>"


def fallback_q_py(question: str):
    qpy = Path(__file__).resolve().parent / 'q.py'
    if not qpy.exists():
        return []
    try:
        proc = subprocess.run([sys.executable, str(qpy)], input=question + "\nexit\n", text=True, capture_output=True, timeout=120)
        out = proc.stdout
        parts = out.split('===== 回答 =====')
        if len(parts) >= 2:
            ans = parts[1].strip()
            return [{'text': ans, 'score': 1.0, 'source': 'FullDocRetriever'}]
        return []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', required=True)
    parser.add_argument('--llm', default='qwen:7b')
    parser.add_argument('--embed-model', default='BAAI/bge-m3')
    args = parser.parse_args()

    q = args.question

    try:
        from sentence_transformers import SentenceTransformer
        from raganything import RAGAnything, RAGAnythingConfig
        from lightrag.utils import EmbeddingFunc

        st = SentenceTransformer(args.embed_model, trust_remote_code=True)

        async def local_embed_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            res = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            return res

        async def llm_model_func(prompt, **kwargs):
            try:
                return await asyncio.to_thread(run_ollama, args.llm, prompt)
            except AttributeError:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, run_ollama, args.llm, prompt)

        embedding_dim = 1024 if hasattr(st, 'get_sentence_embedding_dimension') and st.get_sentence_embedding_dimension() else 768

        embedding_func = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=local_embed_func)

        base_dir = Path(__file__).resolve().parent

        rag_entries = []
        try:
            if RAG_Retriever and isinstance(RAG_Retriever, dict):
                for name, meta in RAG_Retriever.items():
                    if isinstance(meta, dict) and meta.get('path'):
                        rag_entries.append((name, None, Path(meta.get('path'))))
                    else:
                        rag_entries.append((name, meta, None))
        except Exception:
            rag_entries = []

        if not rag_entries:
            try:
                init_FullDocRetriever_stub(str(REPO_ROOT))
            except Exception as e_stub:
                print(f"[FullDocRetriever] init_FullDocRetriever_stub failed: {e_stub}", file=sys.stderr)
            try:
                if RAG_Retriever and isinstance(RAG_Retriever, dict):
                    for name, rag_meta in RAG_Retriever.items():
                        if isinstance(rag_meta, dict) and rag_meta.get('path'):
                            rag_entries.append((name, None, Path(rag_meta.get('path'))))
                        else:
                            rag_entries.append((name, rag_meta, None))
            except Exception:
                rag_entries = []

        if not rag_entries:
            dataset_names = [
                'rag_storage',
                'rag_storage_guanlunji',
                'rag_storage_guide',
                'rag_storage_maintain',
                'rag_storage_qilunji',
                'rag_storage_reportpaper',
                'rag_storage_security',
            ]
            for name in dataset_names:
                p = base_dir / name
                if p.exists():
                    try:
                        config = RAGAnythingConfig(working_dir=str(p), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
                        rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)
                        rag_entries.append((name, rag, p))
                    except Exception as e:
                        print(f"[FullDocRetriever] failed to init {name}: {e}", file=sys.stderr)
                        continue

            if not rag_entries and (base_dir / 'rag_storage').exists():
                p = base_dir / 'rag_storage'
                try:
                    config = RAGAnythingConfig(working_dir=str(p), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
                    rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)
                    rag_entries.append(('rag_storage', rag, p))
                except Exception as e:
                    print(f"[FullDocRetriever] failed to init rag_storage: {e}", file=sys.stderr)

        async def orchestrate_and_query(rag_entries):
            results_local = []
            for entry in rag_entries:
                try:
                    name, rag, p = entry

                    if p is not None:
                        status_file = p / 'kv_store_doc_status.json'
                        try:
                            already_processed = False
                            if status_file.exists():
                                sf = json.load(open(status_file, 'r', encoding='utf-8'))
                                for v in sf.values():
                                    if isinstance(v, dict) and v.get('status') == 'processed':
                                        already_processed = True
                                        break
                        except Exception:
                            already_processed = False

                        has_ready_flag = False
                        try:
                            meta = RAG_Retriever.get(name) if isinstance(RAG_Retriever, dict) else None
                            if isinstance(meta, dict) and meta.get('ready'):
                                has_ready_flag = True
                        except Exception:
                            has_ready_flag = False

                        if not already_processed and not has_ready_flag:
                            try:
                                init_lightrag([(name, p)], embed_model=args.embed_model, llm=args.llm)
                                try:
                                    if isinstance(RAG_Retriever, dict):
                                        RAG_Retriever[name] = {'path': str(p), 'ready': True}
                                except Exception:
                                    pass
                            except Exception as e_init:
                                print(json.dumps({'storage': str(p), 'error': 'init_failed', 'detail': str(e_init)}, ensure_ascii=False), file=sys.stderr)
                                continue

                    if rag is None and p is not None:
                        try:
                            config = RAGAnythingConfig(working_dir=str(p), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
                            rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)
                            try:
                                meta = RAG_Retriever.get(name) if isinstance(RAG_Retriever, dict) else None
                                if isinstance(meta, dict) and meta.get('ready'):
                                    try:
                                        await rag._ensure_lightrag_initialized()
                                    except Exception as e_load:
                                        print(json.dumps({'storage': str(p), 'error': 'load_lightrag_failed', 'detail': str(e_load)}, ensure_ascii=False), file=sys.stderr)
                            except Exception:
                                pass
                        except Exception as e_inst:
                            print(json.dumps({'storage': str(p), 'error': 'instantiation_failed', 'detail': str(e_inst)}, ensure_ascii=False), file=sys.stderr)
                            continue

                    try:
                        if asyncio.iscoroutinefunction(getattr(rag, 'aquery', None)):
                            res = await rag.aquery(q, mode='hybrid')
                        else:
                            res = await asyncio.get_event_loop().run_in_executor(None, lambda: rag.aquery(q, mode='hybrid'))
                    except Exception as e_query:
                        err_s = str(e_query)
                        if 'LightRAG' in err_s or 'lightrag' in err_s or 'No LightRAG' in err_s:
                            try:
                                if p is not None:
                                    init_lightrag([(name, p)], embed_model=args.embed_model, llm=args.llm)
                                if asyncio.iscoroutinefunction(getattr(rag, 'aquery', None)):
                                    res = await rag.aquery(q, mode='hybrid')
                                else:
                                    res = await asyncio.get_event_loop().run_in_executor(None, lambda: rag.aquery(q, mode='hybrid'))
                            except Exception as e_retry:
                                print(json.dumps({'storage': str(p) if p is not None else name, 'error': 'aquery_failed', 'detail': str(e_retry)}, ensure_ascii=False), file=sys.stderr)
                                continue
                        else:
                            print(json.dumps({'storage': str(p) if p is not None else name, 'error': 'aquery_failed', 'detail': str(e_query)}, ensure_ascii=False), file=sys.stderr)
                            continue

                    try:
                        if isinstance(res, dict):
                            text = res.get('answer') or res.get('text') or json.dumps(res, ensure_ascii=False)
                        else:
                            text = str(res)
                    except Exception:
                        text = str(res)
                    results_local.append({'text': text, 'score': 1.0, 'source': name})
                except Exception as e:
                    print(f"[FullDocRetriever] query failed for {entry}: {e}", file=sys.stderr)
                    continue
            return results_local

        results = asyncio.run(orchestrate_and_query(rag_entries))
        if results:
            def _norm_text(t: str) -> str:
                try:
                    return ' '.join(str(t).split()).lower()
                except Exception:
                    return str(t)

            dedup = {}
            for c in results:
                text = c.get('text', '')
                key = _norm_text(text)
                score = float(c.get('score', 1.0)) if c.get('score') is not None else 1.0
                src = c.get('source')
                if key not in dedup:
                    dedup[key] = {'text': text, 'score': score, 'sources': [src] if src else [], 'raw': c}
                else:
                    if score > dedup[key]['score']:
                        dedup[key]['text'] = text
                        dedup[key]['score'] = score
                        dedup[key]['raw'] = c
                    if src and src not in dedup[key]['sources']:
                        dedup[key]['sources'].append(src)

            deduped_results = []
            for v in dedup.values():
                item = v.get('raw', {'text': v['text'], 'score': v['score']})
                if v['sources']:
                    item['source'] = ','.join([s for s in v['sources'] if s])
                deduped_results.append(item)

            print(json.dumps({'candidates': deduped_results}, ensure_ascii=False))
            return
        if not results:
            cand = fallback_q_py(q)
            print(json.dumps({'candidates': cand}, ensure_ascii=False))
            return
    except Exception:
        cand = fallback_q_py(q)
        print(json.dumps({'candidates': cand}, ensure_ascii=False))


if __name__ == '__main__':
    main()
