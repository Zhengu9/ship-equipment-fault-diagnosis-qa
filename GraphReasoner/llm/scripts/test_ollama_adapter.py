#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import subprocess

class OllamaInline:
    def __init__(self, model_name='tinyllama'):
        # support passing either a string or an args-like object with attribute `ollama_model`
        if hasattr(model_name, 'ollama_model'):
            self.model_name = getattr(model_name, 'ollama_model')
        else:
            self.model_name = model_name

    def prepare_for_inference(self):
        return

    def generate_sentence(self, prompt: str):
        try:
            proc = subprocess.run(['ollama', 'run', self.model_name], input=prompt, text=True, capture_output=True, timeout=120)
            out = proc.stdout.strip()
            if out:
                return out
            return proc.stderr.strip()
        except Exception as e:
            return str(e)

Ollama = OllamaInline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tinyllama', help='ollama model name')
    parser.add_argument('--prompt', default='Hello, who won the world cup in 2018?', help='prompt')
    args = parser.parse_args()

    # create a fake args namespace similar to predict_answer
    class A:
        pass

    a = A()
    a.ollama_model = args.model

    model = Ollama(a)
    model.prepare_for_inference()
    out = model.generate_sentence(args.prompt)
    print('=== OUTPUT ===')
    print(out)

if __name__ == '__main__':
    main()
