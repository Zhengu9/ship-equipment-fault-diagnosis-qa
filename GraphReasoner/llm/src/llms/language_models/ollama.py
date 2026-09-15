import subprocess
from .base_language_model import BaseLanguageModel


class Ollama(BaseLanguageModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--ollama_model', type=str, default='tinyllama', help='ollama model name to run')

    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096

    def prepare_for_inference(self, **model_kwargs):
        # nothing to load locally besides checking ollama availability
        pass

    def tokenize(self, text):
        return len(text.split())

    def generate_sentence(self, llm_input):
        try:
            # pass prompt via stdin to ollama run
            proc = subprocess.run([
                'ollama', 'run', self.args.ollama_model
            ], input=llm_input, text=True, capture_output=True, timeout=120)
            out = proc.stdout.strip()
            if out:
                return out
            # if no stdout, return stderr for debugging
            return proc.stderr.strip()
        except Exception as e:
            return str(e)
