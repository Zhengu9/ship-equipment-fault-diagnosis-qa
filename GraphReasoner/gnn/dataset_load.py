import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
import random
import warnings
import pickle
warnings.filterwarnings("ignore")
from gnn.modules.question_encoding.tokenizers import LSTMTokenizer#, BERTTokenizer
from transformers import AutoTokenizer
import time

import os


class BasicDataLoader(object):

    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        self.tokenize = tokenize
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()
        

    def _load_file(self, config, data_type="train"):

        data_file = config['data_folder'] + data_type + ".json"
        self.data_file = data_file
        print('loading data from', data_file)
        self.data_type = data_type
        self.data = []
        skip_index = set()
        index = 0

        with open(data_file) as f_in:
            for line in tqdm(f_in):
                if index == config['max_train'] and data_type == "train": break  #break if we reach max_question_size
                line = json.loads(line)
                
                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                index += 1

        print("skip", skip_index)
        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _load_data(self):

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        self._prepare_data()

    def _parse_args(self, config, word2id, relation2id, entity2id):

        self.data_eff = config['data_eff']
        self.data_name = config['name']

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        self.rel_word_emb = config['relation_word_emb']
        #self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        print('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        self.q_type = config['q_type']

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    
    def get_quest(self, training=False):
        q_list = []
        
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            # id2word = self.id2word
            # for i in range(self.max_query_word):
            #     if self.query_texts[sample_id, i] in id2word:
            #         tp_str += id2word[self.query_texts[sample_id, i]] + " "
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        if self.tokenize == 'lstm':
            id2word = self.id2word
            tp_str = ""
            for i in range(self.max_query_word):
                if np_array_x[i] in id2word:
                    tp_str += id2word[np_array_x[i]] + " "
        else:
            tp_str = ""
            try:
                ids = [int(x) for x in np_array_x]
                # remove sentinel pad id values if present
                ids = [i for i in ids if i != self.num_word]
                tp_str = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            except Exception:
                # fallback to token-level conversion for older tokenizers
                words = self.tokenizer.convert_ids_to_tokens(np_array_x)
                for w in words:
                    if w not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                        tp_str += w + " "
        return tp_str
    

    def _prepare_data(self):

        max_count = 0

        if self.tokenize == 'lstm':
            for line in self.data:
                word_list = line["question"].split(' ')
                max_count = max(max_count, len(word_list))
        else:

            if self.tokenize == 'bert':
                tmp_tokenizer_name = 'bert-base-chinese'
            elif self.tokenize == 'roberta':
                tmp_tokenizer_name = 'roberta-base'
            elif self.tokenize == 'xlm':
                tmp_tokenizer_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'xlm-roberta-base')
            elif self.tokenize == 'sbert':
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                local_sbert = os.path.join(repo_root, 'pretrained_lms', 'all-MiniLM-L6-v2')
                local_sr = os.path.join(repo_root, 'pretrained_lms', 'sr-simbert')
                if os.path.isdir(local_sbert):
                    tmp_tokenizer_name = local_sbert
                elif os.path.isdir(local_sr):
                    tmp_tokenizer_name = local_sr
                else:
                    tmp_tokenizer_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            elif self.tokenize == 'sbert2':
                tmp_tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif self.tokenize == 't5':
                tmp_tokenizer_name = 't5-small'
            elif self.tokenize == 'simcse':
                tmp_tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif self.tokenize == 'relbert':
                tmp_tokenizer_name = 'pretrained_lms/sr-simbert/'
            else:
                tmp_tokenizer_name = 'bert-base-chinese'

            try:
                tmp_tok = AutoTokenizer.from_pretrained(tmp_tokenizer_name)
            except Exception:
                tmp_tok = AutoTokenizer.from_pretrained(tmp_tokenizer_name, local_files_only=True)

            for line in self.data:
                toks = tmp_tok.tokenize(line["question"])
                max_count = max(max_count, len(toks))

        
        if self.rel_word_emb:
            self.build_rel_words(self.tokenize)
        else:
            self.rel_texts = None
            self.rel_texts_inv = None
            self.ent_texts = None



        self.max_query_word = max_count
        #self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        #self.query_texts2 = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)

        #build tokenizers
        if self.tokenize == 'lstm':
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)
        else:
            if self.tokenize == 'bert':
                tokenizer_name = 'bert-base-chinese'    
            elif self.tokenize  == 'roberta':
                tokenizer_name = 'roberta-base'
            elif self.tokenize == 'xlm':
                tokenizer_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'xlm-roberta-base')
            elif self.tokenize  == 'sbert':
                # Prefer local pretrained model if available (offline use)
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                local_sbert = os.path.join(repo_root, 'pretrained_lms', 'all-MiniLM-L6-v2')
                local_sr = os.path.join(repo_root, 'pretrained_lms', 'sr-simbert')
                if os.path.isdir(local_sbert):
                    tokenizer_name = local_sbert
                elif os.path.isdir(local_sr):
                    tokenizer_name = local_sr
                else:
                    tokenizer_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            elif self.tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize  == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'

            self.max_query_word = max_count + 2 #for cls token and sep
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

            # Cap max_query_word to the tokenizer/model maximum to avoid exceeding
            # the LM's positional embedding size (prevents indexing errors).
            try:
                tok_max = getattr(self.tokenizer, 'model_max_length', None)
            except Exception:
                tok_max = None
            if tok_max and self.max_query_word > tok_max:
                print(f"Warning: max_query_word ({self.max_query_word}) > tokenizer.model_max_length ({tok_max}), capping.")
                self.max_query_word = int(tok_max)

            # num_word is used as pad id sentinel
            if getattr(self.tokenizer, 'pad_token_id', None) is not None:
                self.num_word = int(self.tokenizer.pad_token_id)
            else:
                try:
                    pid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                    if pid is None:
                        raise Exception
                    self.num_word = int(pid)
                except Exception:
                    self.num_word = len(self.word2id)

            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)


        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            #print(g2l)
            if len(g2l) == 0:
                #print(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            key_ent = 'entities_cid' if 'entities_cid' in sample else 'entities'
            for j, entity in enumerate(sample[key_ent]):
                # Resolve global entity id from entity object
                global_entity = None
                try:
                    if isinstance(entity, dict):
                        # prefer kb_id if present, otherwise use text
                        if 'kb_id' in entity and entity['kb_id'] in self.entity2id:
                            global_entity = self.entity2id[entity['kb_id']]
                        elif 'text' in entity and entity['text'] in self.entity2id:
                            global_entity = self.entity2id[entity['text']]
                        else:
                            # fallback: try both keys as-is in mapping
                            if 'kb_id' in entity and entity['kb_id'] in self.entity2id:
                                global_entity = self.entity2id[entity['kb_id']]
                            elif 'text' in entity and entity['text'] in self.entity2id:
                                global_entity = self.entity2id[entity['text']]
                            else:
                                raise KeyError('entity not in entity2id')
                    else:
                        global_entity = self.entity2id[entity]
                except Exception:
                    # if mapping failed, keep raw value (assumed already a global id)
                    global_entity = entity

                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if self.data_name != 'cwq':

                    if local_entity not in tp_set:  # skip entities in question
                    #print(global_entity)
                    #print(local_entity)
                        self.candidate_entities[next_id, local_entity] = global_entity
                elif self.data_name == 'cwq':
                    self.candidate_entities[next_id, local_entity] = global_entity
                # if local_entity != 0:  # skip question node
                #     self.candidate_entities[next_id, local_entity] = global_entity

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                try:
                    if isinstance(sbj, dict) and  'text' in sbj:
                        head = g2l[self.entity2id[sbj['text']]]
                        rel = self.relation2id[rel['text']]
                        tail = g2l[self.entity2id[obj['text']]]
                    else:
                        head = g2l[self.entity2id[sbj]]
                        rel = self.relation2id[rel]
                        tail = g2l[self.entity2id[obj]]
                except:
                    head = g2l[sbj]
                    try:
                        rel = int(rel)
                    except:
                        rel = self.relation2id[rel]
                    tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                self.kb_fact_rels[next_id, i] = rel
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
                    self.kb_fact_rels[next_id, i] = rel + len(self.relation2id)
                
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                print(next_id, len(tp_set))
                exit(-1)

            #tokenize question
            if self.tokenize == 'lstm':
                self.query_texts[next_id] = self.tokenizer.tokenize(sample['question'])
            else:
                tokens = self.tokenizer(text=sample['question'], max_length=self.max_query_word, 
                                        padding='max_length', return_attention_mask=False, truncation=True, return_tensors='pt')
                self.query_texts[next_id] = np.array(tokens['input_ids'][0].tolist())


            # construct distribution for answers
            answer_list = []
            if 'answers_cid' in sample:
                for answer in sample['answers_cid']:
                    # Normalize different possible formats of answers_cid to a global entity id (int)
                    global_id = None
                    # case 1: answer is an int (could be an index into sample['entities'] or already a global id)
                    if isinstance(answer, int):
                        # if entities list exists and index in range, map to that entity's global id when possible
                        if 'entities' in sample and isinstance(sample['entities'], list) and answer < len(sample['entities']):
                            ent_repr = sample['entities'][answer]
                            if isinstance(ent_repr, dict):
                                key = 'kb_id' if 'kb_id' in ent_repr else ('text' if 'text' in ent_repr else None)
                                val = ent_repr[key] if key is not None else ent_repr
                            else:
                                val = ent_repr
                            try:
                                global_id = self.entity2id[val]
                            except Exception:
                                # val might already be a numeric global id
                                global_id = val
                        else:
                            # treat as already a global numeric id
                            global_id = answer
                    else:
                        # answer may be a string id or an object
                        if isinstance(answer, dict):
                            key = 'kb_id' if 'kb_id' in answer else ('text' if 'text' in answer else None)
                            val = answer[key] if key is not None else answer
                        else:
                            val = answer
                        try:
                            global_id = self.entity2id[val]
                        except Exception:
                            global_id = val

                    answer_list.append(global_id)
                    if global_id in g2l:
                        self.answer_dists[next_id, g2l[global_id]] = 1.0
            else:
                for answer in sample['answers']:
                    # prefer kb_id if present, otherwise use text
                    keyword = 'kb_id' if 'kb_id' in answer else 'text'
                    try:
                        answer_ent = self.entity2id[answer[keyword]]
                    except Exception:
                        answer_ent = answer[keyword]
                    answer_list.append(answer_ent)
                    if answer_ent in g2l:
                        self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list

            if not self.data_eff:
                self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        print("{} cases in total, {} cases without query entity, {} cases with single query entity,"
              " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                              num_one_query_ent, num_multiple_ent))

        
    def build_rel_words(self, tokenize):

        max_rel_words = 0
        rel_words = []
        if 'metaqa' in self.data_file:
            for rel in self.relation2id:
                words = rel.split('_')
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
            #print(rel_words)
        else:
            for rel in self.relation2id:
                rel = rel.strip()
                fields = rel.split('.')
                try:
                    words = fields[-2].split('_') + fields[-1].split('_')
                    max_rel_words = max(len(words), max_rel_words)
                    rel_words.append(words)
                    #print(rel, words)
                except:
                    words = ['UNK']
                    rel_words.append(words)
                    pass
                #words = fields[-2].split('_') + fields[-1].split('_')
            
        self.max_rel_words = max_rel_words
        if tokenize == 'lstm':
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            for rel_id,tokens in enumerate(rel_words):
                for j, word in enumerate(tokens):
                    if j < self.max_rel_words:
                            if word in self.word2id:
                                self.rel_texts[rel_id, j] = self.word2id[word]
                                self.rel_texts_inv[rel_id, j] = self.word2id[word]
                            else:
                                self.rel_texts[rel_id, j] = len(self.word2id)
                                self.rel_texts_inv[rel_id, j] = len(self.word2id)
        else:
            if tokenize == 'bert':
                tokenizer_name = 'bert-base-chinese'
            elif tokenize == 'roberta':
                tokenizer_name = 'roberta-base'
            elif tokenize == 'xlm':
                tokenizer_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'xlm-roberta-base')
            elif tokenize == 'sbert':
                # Prefer local pretrained model if available (offline use)
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                local_sbert = os.path.join(repo_root, 'pretrained_lms', 'all-MiniLM-L6-v2')
                local_sr = os.path.join(repo_root, 'pretrained_lms', 'sr-simbert')
                if os.path.isdir(local_sbert):
                    tokenizer_name = local_sbert
                elif os.path.isdir(local_sr):
                    tokenizer_name = local_sr
                else:
                    tokenizer_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            elif tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif tokenize == 't5':
                tokenizer_name = 't5-small'
            elif tokenize  == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            
            for rel_id,words in enumerate(rel_words):

                # use tokenizer(...) which works across transformers versions
                # request tensor output to ensure a consistent dict with 'input_ids'
                tokens = tokenizer(text=' '.join(words), max_length=self.max_rel_words, 
                                   padding='max_length', return_attention_mask=False, truncation=True, return_tensors='pt')
                tokens_inv = tokenizer(text=' '.join(words[::-1]), max_length=self.max_rel_words, 
                                       padding='max_length', return_attention_mask=False, truncation=True, return_tensors='pt')
                # fallback if tokenizer returns nested dicts
                # extract input_ids robustly for different tokenizer return types
                # tokens now have tensor values; extract lists
                try:
                    ids = tokens['input_ids'][0].tolist()
                    ids_inv = tokens_inv['input_ids'][0].tolist()
                except Exception:
                    raise RuntimeError('Unable to extract input_ids from tokenizer output (tensor extraction failed)')
                self.rel_texts[rel_id] = np.array(ids)
                self.rel_texts_inv[rel_id] = np.array(ids_inv)


        
        #print(rel_words)
        #print(len(rel_words), len(self.relation2id))
        assert len(rel_words) == len(self.relation2id)
        #print(self.rel_texts, self.max_rel_words)

    def create_kb_adj_mats(self, sample_id):

        sample = self.data[sample_id]
        g2l = self.global2local_entity_maps[sample_id]
        
        # build connection between question and entities in it
        head_list = []
        rel_list = []
        tail_list = []
        for i, tpl in enumerate(sample['subgraph']['tuples']):
            sbj, rel, obj = tpl
            try:
                if isinstance(sbj, dict) and  'text' in sbj:
                    head = g2l[self.entity2id[sbj['text']]]
                    rel = self.relation2id[rel['text']]
                    tail = g2l[self.entity2id[obj['text']]]
                else:
                    head = g2l[self.entity2id[sbj]]
                    rel = self.relation2id[rel]
                    tail = g2l[self.entity2id[obj]]
            except:
                head = g2l[sbj]
                try:
                    rel = int(rel)
                except:
                    rel = self.relation2id[rel]
                tail = g2l[obj]
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            if self.use_inverse_relation:
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)

        return np.array(head_list, dtype=int),  np.array(rel_list, dtype=int), np.array(tail_list, dtype=int)

    
    def _build_fact_mat(self, sample_ids, fact_dropout):

        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        #print(sample_ids)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            if self.data_eff:
                head_list, rel_list, tail_list = self.create_kb_adj_mats(sample_id) #kb_adj_mats[sample_id]
            else:
                (head_list, rel_list, tail_list) = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_rels_ids = zip(batch_heads, batch_rels)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]

        
        head_rels_batch = list(zip(batch_heads, batch_rels))
        #print(head_rels_batch)
        head_rels_count = Counter(head_rels_batch)
        weight_rel_list = [1.0 / head_rels_count[(h,r)] for (h,r) in head_rels_batch]

        #print(head_rels_count)

        # tail_count = Counter(batch_tails)

        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list


    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):

        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            # If entities_cid exists and appears to be indices into sample['entities'], resolve them
            if 'entities_cid' in sample:
                ecs = sample['entities_cid']
                # If sample['entities'] is a list and ecs contains integer indices, map to actual entity representations
                if isinstance(sample.get('entities'), list) and len(sample['entities']) > 0 and isinstance(ecs[0], int):
                    resolved = []
                    for cid in ecs:
                        try:
                            ent_repr = sample['entities'][cid]
                        except Exception:
                            ent_repr = cid
                        # normalize dict entries to kb_id/text or raw value
                        if isinstance(ent_repr, dict):
                            key = 'kb_id' if 'kb_id' in ent_repr else ('text' if 'text' in ent_repr else None)
                            val = ent_repr[key] if key is not None else ent_repr
                        else:
                            val = ent_repr
                        resolved.append(val)
                    self._add_entity_to_map(self.entity2id, resolved, g2l)
                else:
                    self._add_entity_to_map(self.entity2id, ecs, g2l)
            else:
                self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            #self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps



    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        #print(entities)
        #print(entity2id)
        for entity_global_id in entities:
            try:
                # handle dict representations with kb_id or text
                if isinstance(entity_global_id, dict):
                    key = 'kb_id' if 'kb_id' in entity_global_id else ('text' if 'text' in entity_global_id else None)
                    if key is not None:
                        lookup = entity_global_id[key]
                        try:
                            ent = entity2id[lookup]
                        except Exception:
                            ent = lookup
                    else:
                        ent = entity_global_id
                else:
                    try:
                        ent = entity2id[entity_global_id]
                    except Exception:
                        ent = entity_global_id

                if ent not in g2l:
                    g2l[ent] = len(g2l)
            except Exception:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        
        return q_input

    



class SingleDataLoader(BasicDataLoader):

    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_dict_int(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[int(word)] = int(word)
    return word2id

def load_data(config, tokenize):

    if 'sr-cwq' in config['data_folder']:
        entity2id = load_dict_int(config['data_folder'] + config['entity2id'])
    else:
        entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    
    if config["is_eval"]:
        train_data = None
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = test_data.num_word
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = train_data.num_word
    relation_texts = test_data.rel_texts
    relation_texts_inv = test_data.rel_texts_inv
    entities_texts = None
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data, #test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": num_word,
        "rel_texts": relation_texts,
        "rel_texts_inv": relation_texts_inv,
        "ent_texts": entities_texts
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    #args = get_config()
    load_data(args)
