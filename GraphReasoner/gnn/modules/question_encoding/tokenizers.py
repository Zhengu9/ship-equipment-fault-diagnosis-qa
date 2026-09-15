import re
import numpy as np
from transformers import BertTokenizer

class LSTMTokenizer():
    def __init__(self, word2id, max_query_word):
        super(LSTMTokenizer, self).__init__()
        self.word2id = word2id
        self.max_query_word = max_query_word

    def tokenize(self, question):
        tokens = self.tokenize_sent(question)
        query_text = np.full(self.max_query_word, len(self.word2id), dtype=int)
        #tokens = question.split()
        #if self.data_type == "train":
        #    random.shuffle(tokens)
        for j, word in enumerate(tokens):
            if j < self.max_query_word:
                    if word in self.word2id:
                        query_text[j] = self.word2id[word]
                        
            else:
                query_text[j] = len(self.word2id)

        return query_text

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip()
        # 保留中文、字母和数字序列；将英文小写化以匹配 vocab
        question_text = question_text.replace("'s", ' s')
        # 提取中文连续字符或英数字连续字符
        tokens = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', question_text)
        # 英文小写
        tokens = [t.lower() if re.match(r'[A-Za-z0-9]+', t) else t for t in tokens]
        return tokens

class BERTTokenizer():
    def __init__(self, max_query_word):
        super(BERTTokenizer, self).__init__()
        self.q_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_query_word = max_query_word
        self.num_word = self.q_tokenizer.encode("[UNK]")[0] #len(self.q_tokenizer.vocab.keys())

        
    
    def tokenize(self, question):
        query_text = np.full(self.max_query_word, 0, dtype=int)
        tokens = self.q_tokenizer(text=question, max_length=self.max_query_word, 
                      padding='max_length', return_attention_mask=False, truncation=True, return_tensors='pt')
        return np.array(tokens['input_ids'][0].tolist())