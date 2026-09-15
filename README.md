# 船舶装备故障诊断智能问答系统

面向船舶机电设备维护与故障诊断的本地化智能问答系统。项目以检索增强生成（RAG）为主线，结合船舶维修资料、知识图谱推理和语义对齐能力，为设备故障排查、维修操作和应急处置提供问答支持。

## 功能概览

- 基于 LightRAG/RAGAnything 的多知识库检索问答
- 支持主机、辅机、电气、液压、泵阀、锅炉、制冷和应急操作等船舶装备主题
- 支持 Ollama 本地大语言模型，例如 `qwen:7b`
- 使用 `BAAI/bge-m3` 进行文本向量化
- 提供 GraphReasoner 图推理与可解释问答实验代码
- 提供 SemanticAligner 语义对齐、Milvus 向量检索和 ShipQA 流程
- 支持将问答历史导出为 XLSX 文件

## 目录结构

```text
.
├── rag_cli.py                 # 根目录交互式问答入口
├── init_rags.py               # 知识库初始化与 Ollama/Milvus 辅助逻辑
├── FullDocRetriever/          # 多文档 RAG 检索实现
├── GraphReasoner/             # 知识图谱推理与图神经网络实验
├── SemanticAligner/           # 语义对齐、Milvus 与 ShipQA 流程
├── requirements.txt           # 根目录运行依赖
├── .env.example               # 环境变量示例
└── .gitignore                 # 本地模型、缓存和运行产物忽略规则
```

## 环境要求

- Linux/WSL2，推荐 Ubuntu 22.04 或更新版本
- Python 3.10+
- 可用的 NVIDIA GPU 与 CUDA 环境（部分模型和 GraphReasoner 实验需要）
- Ollama
- Docker Desktop 或 Docker Engine（SemanticAligner 的 Milvus 流程需要）
- Git LFS（如果你计划单独管理大型模型或数据集）

## 快速开始

### 1. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell 可使用：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 准备本地模型

安装并启动 Ollama，然后准备问答模型：

```bash
ollama pull qwen:7b
ollama serve
```

首次运行时，`BAAI/bge-m3` 会由 `sentence-transformers` 下载。也可以提前下载到本地，并通过 `EMBED_MODEL` 或代码参数指定模型路径。

复制环境变量模板：

```bash
cp .env.example .env
```

如需访问 Hugging Face 的受限模型，在 `.env` 中填写 `HF_TOKEN`。不要把真实 token 写入代码、提交到 Git 或发布到公开仓库。

### 3. 启动交互式问答

在项目根目录运行：

```bash
python rag_cli.py
```

初始化完成后，在 `问题：` 提示符输入问题。输入 `q`、`quit` 或 `exit` 退出；输入 `output` 可将 `QAhistory.jsonl` 导出为 XLSX。

也可以直接执行指定问题：

```bash
python FullDocRetriever/query_single.py --question "主柴油机启动失败应该如何排查？"
```

## SemanticAligner

启动 Milvus 依赖服务：

```bash
cd SemanticAligner/pipeline
docker compose up -d
```

然后根据 `SemanticAligner/configs/ShipQA.json` 准备 ShipQA 数据和模型，执行相应索引与查询脚本：

```bash
python SemanticAligner/pipeline/ShipQA_index.py
python SemanticAligner/pipeline/query_single.py
```

Milvus 数据卷位于 `SemanticAligner/pipeline/volumes/`，属于本地运行产物，不应提交到 Git。

## GraphReasoner

GraphReasoner 使用独立的实验环境，依赖版本与根目录问答环境不同。推荐按 `GraphReasoner/environment.yml` 或 `GraphReasoner/gnn/requirements.txt` 创建单独环境，再根据脚本参数准备数据集、预训练模型和 GPU 环境。

相关入口包括：

- `GraphReasoner/gnn/main.py`：图神经网络训练入口
- `GraphReasoner/gnn/evaluate.py`：评估入口
- `GraphReasoner/llm/src/qa_prediction/predict_answer.py`：基于语言模型的答案预测
- `GraphReasoner/llm/src/qa_prediction/gen_rule_path.py`：关系路径生成

## 数据与生成文件

为避免提交大型文件、用户资料和运行时缓存，以下内容默认被 `.gitignore` 忽略：

- `FullDocRetriever/rag_storage*` 和解析输出
- `SemanticAligner/pipeline/volumes`
- 模型权重、检查点和本地模型目录
- `GraphReasoner` 的 `checkpoint`、`results` 和数据目录
- `.idea`、虚拟环境、日志和问答历史

克隆仓库后，需要根据自己的资料重新生成索引，并将数据集放在对应目录。项目不会在仓库中提供船舶资料原文或本地模型权重。

## 配置说明

主要默认配置如下：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen:7b` | Ollama 对话模型 |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI 兼容接口 |
| `EMBED_MODEL` | `BAAI/bge-m3` | 文本嵌入模型 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face 下载源，可按网络环境调整 |
| `HF_TOKEN` | 空 | 受限 Hugging Face 模型访问令牌 |

部分脚本仍支持命令行参数覆盖默认模型和路径，具体以脚本的 `--help` 输出为准。

## 安全提示

- 不要提交 `.env`、访问令牌、密码、服务器部署配置或云端密钥。
- 生产环境请替换示例中的默认服务密码，并限制 Milvus、MinIO 等服务的网络暴露范围。
- 本项目输出仅供辅助分析，实际维修与应急操作应以船舶设备手册、船级社规范和现场安全规程为准。

## 许可证

当前仓库未声明开源许可证。若计划公开发布，请根据原始代码、数据集、模型和第三方依赖的授权情况补充许可证及版权说明。
