#!/usr/bin/env python3
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Local model folder (Llama-2-7b-chat-hf)')
    parser.add_argument('--prompt', default='Hello, what is the capital of France?', help='Test prompt')
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        print('缺少依赖，请先在环境中安装: pip install transformers torch huggingface_hub')
        print('错误:', e)
        sys.exit(1)

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        print('找不到指定模型目录：', model_dir)
        sys.exit(1)

    print('加载分词器...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    except Exception as e:
        print('分词器加载失败:', e)
        sys.exit(1)

    print('尝试加载模型（可能需要大量内存）...')
    model = None
    try:
        # 首选自动映射到可用设备（GPU）并使用 fp16 加速；失败则回退到 CPU 加载
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print('自动设备映射加载失败，尝试 CPU 加载（可能很慢且内存不足）:', e)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={'': 'cpu'}, trust_remote_code=True)
        except Exception as e2:
            print('CPU 加载也失败:', e2)
            sys.exit(1)

    print('模型加载成功。进行一次小示例生成（最多 32 个新 token）...')
    try:
        input_ids = tokenizer(args.prompt, return_tensors='pt').input_ids
        # 将输入移动到模型设备
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        gen = model.generate(input_ids, max_new_tokens=32)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print('生成结果:')
        print(out)
    except Exception as e:
        print('生成时出错（资源/兼容性问题常见）:', e)

if __name__ == '__main__':
    main()
