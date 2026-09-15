from openai import OpenAI


class LLM:
    def __init__(self, configs):
        self.configs = configs['llm']
        print('Loading LLM ...')
        self.client = OpenAI(
            base_url = self.configs['base_url'],
            api_key = self.configs['api_key']
        )

    def chat(self, input_text):
        completion = self.client.chat.completions.create(
            model=self.configs['model'],
            messages=[{"role":"user", "content":input_text}],
            temperature=self.configs['temperature'],
            top_p=self.configs['top_p'],
            max_tokens=self.configs['max_tokens']
        )
        return completion.choices[0].message.content.strip()