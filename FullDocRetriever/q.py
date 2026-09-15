import os
import asyncio
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen:7b"


EMBED_MODEL_PATH = "BAAI/bge-m3"

os.environ["OMP_NUM_THREADS"] = "1"


embed_model = SentenceTransformer(
    EMBED_MODEL_PATH,
    device="cpu",
)


async def local_embed_func(texts):
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    vectors = embed_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vectors



def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def main():
    config = RAGAnythingConfig(
        working_dir=str(REPO_ROOT / 'Anything' / 'rag_storage_guide'),
        parser="mineru",
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=local_embed_func,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    print("初始化 LightRAG...")

    init_file = REPO_ROOT / 'Anything' / 'init.txt'


    if not init_file.exists():
        init_file.write_text('init', encoding='utf-8')

    await rag.process_document_complete(
        file_path=str(init_file),
        output_dir=str(REPO_ROOT / 'Anything' / 'output'),
        parse_method="auto",
    )

    print("初始化完成")

    while True:
        question = input("\n请输入问题（输入 exit 退出）: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        try:
            result = await rag.aquery(
                question,
                mode="hybrid",
                enable_rerank=False,
            )

            print("\n===== 回答 =====")
            print(result)

        except Exception as e:
            print("\n 查询失败：", str(e))


if __name__ == "__main__":
    asyncio.run(main())


def create_or_init_rag_sync(working_dir: str, llm_model_func, embedding_func):

    async def _do():
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

        wk = Path(working_dir)
        status_file = wk / 'kv_store_doc_status.json'
        has_processed = False
        try:
            if status_file.exists():
                sf = json.load(open(status_file, 'r', encoding='utf-8'))
                for v in sf.values():
                    if isinstance(v, dict) and v.get('status') == 'processed':
                        has_processed = True
                        break
        except Exception:
            has_processed = False

        print(f"has_processed={has_processed} for {working_dir}")

        init_file = REPO_ROOT / 'Anything' / 'init.txt'
        out_dir = REPO_ROOT / 'Anything' / 'output'
        if not init_file.exists():
            init_file.write_text('init', encoding='utf-8')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        await rag.process_document_complete(file_path=str(init_file), output_dir=str(out_dir), parse_method='auto')

        return rag, has_processed

    return asyncio.run(_do())


def run_query_sync(working_dir: str, question: str, llm_model_func, embedding_func):

    async def _do_query():
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

        init_file = REPO_ROOT / 'Anything' / 'init.txt'
        out_dir = REPO_ROOT / 'Anything' / 'output'
        if not init_file.exists():
            init_file.write_text('init', encoding='utf-8')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        await rag.process_document_complete(file_path=str(init_file), output_dir=str(out_dir), parse_method='auto')

        res = await rag.aquery(question, mode='hybrid')
        return res

    return asyncio.run(_do_query())
