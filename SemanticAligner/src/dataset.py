import os
import re
import pickle
import joblib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class Dataset:
    configs = None
    KG = None
    type_to_nodes = None
    queries = None
    groundtruths = None
    
    def __init__(self, configs):
        self.configs = configs
        self._load_data()
        
    def _load_data(self):
        raise NotImplementedError
    
    def get_KG(self):
        return self.KG
    
    def get_type_to_nodes(self):
        return self.type_to_nodes
    
    def get_queries(self):
        return self.queries
    
    def get_groundtruths(self):
        return self.groundtruths
    
    
class MetaQA(Dataset):
    def __init__(self, configs):
        self.hop = configs['hop']
        super().__init__(configs)
        
    def _load_data(self):
        raw_data_dir = self.configs['raw_data_dir']
        processed_data_dir = self.configs['processed_data_dir']
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # load KG
        print("Loading KG ...")
        if os.path.exists(f"{processed_data_dir}/graph"):
            self.KG = pickle.load(open(f"{processed_data_dir}/graph", "rb"))
        else:
            self.KG = {}
            with open(f"{raw_data_dir}/kb.txt", "r", encoding="utf-8") as file:
                for line in file:
                    head, relation, tail = line.strip().split('|')
                    if head not in self.KG:
                        self.KG[head] = {}
                    if tail not in self.KG:
                        self.KG[tail] = {}
                    if relation not in self.KG[head]:
                        self.KG[head][relation] = [tail]
                    else:
                        self.KG[head][relation].append(tail)
                    if relation not in self.KG[tail]:
                        self.KG[tail][relation] = [head]
                    else:
                        self.KG[tail][relation].append(head)
            for head in self.KG:
                for relation in self.KG[head]:
                    self.KG[head][relation] = list(set(self.KG[head][relation]))
            pickle.dump(self.KG, open(f"{processed_data_dir}/graph", "wb"))
        print("- done")
            
        # load queries and groundtruths
        print("Loading queries and groundtruths ...")
        if self.hop == 1:
            qa_filename = f"{raw_data_dir}/1-hop/vanilla/qa_test.txt"
        else:
            qa_filename = f"{raw_data_dir}/{self.hop}-hop/{self.hop}-hop/vanilla/qa_test.txt"
        self.queries, self.groundtruths = [], []
        with open(qa_filename, "r", encoding="utf-8") as file:
            for line in file:
                query, groundtruth = line.strip().split('\t')
                query = query.replace('[', '').replace(']', '')
                groundtruth = groundtruth.split('|')
                self.queries.append(query)
                self.groundtruths.append(groundtruth)
        print("- done")

        
class FactKG(Dataset):
    def __init__(self, configs):
        super().__init__(configs)
        
    def _load_data(self):
        raw_data_dir = self.configs['raw_data_dir']
        processed_data_dir = self.configs['processed_data_dir']
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # load KG
        print("Loading KG ...")
        if os.path.exists(f"{processed_data_dir}/FactKG.graph"):
            self.KG = pickle.load(open(f"{processed_data_dir}/FactKG.graph", "rb"))
            self.type_to_nodes = pickle.load(open(f"{processed_data_dir}/FactKG.type_to_nodes", "rb"))
        else:
            with open(f'{raw_data_dir}/dbpedia_2015_undirected.pickle', 'rb') as f:
                dbpedia = pickle.load(f)
            entities = list(dbpedia)
            with open(f'{raw_data_dir}/relations_for_final.pickle', 'rb') as f:
                relations = set(pickle.load(f))
            
            def process_entity_batch(start, end):
                batch_edges = []
                type_to_nodes = defaultdict(set)
                for entity in entities[start:end]:
                    for relation in dbpedia[entity]:
                        if relation == '22-rdf-syntax-ns#type':
                            for type_name in dbpedia[entity][relation]:
                                if not re.match(r"^Q\d+$", type_name):
                                    type_to_nodes[type_name].add(entity)
                        elif relation in relations and'~' not in relation:
                            for neighbor in dbpedia[entity][relation]:
                                batch_edges.append((entity, neighbor, relation))
                return batch_edges, type_to_nodes

            batchsize=8192
            results = joblib.Parallel(n_jobs=64, backend='threading')(
                joblib.delayed(process_entity_batch)(i, i+batchsize) for i in tqdm(range(0, len(entities), batchsize))
            )

            self.KG = {}
            self.type_to_nodes = defaultdict(set)
            for batch_edges, batch_type_to_nodes in tqdm(results):
                for head, tail, relation in batch_edges:
                    if head not in self.KG:
                        self.KG[head] = {}
                    if tail not in self.KG:
                        self.KG[tail] = {}
                    if relation not in self.KG[head]:
                        self.KG[head][relation] = [tail]
                    else:
                        self.KG[head][relation].append(tail)
                    if relation not in self.KG[tail]:
                        self.KG[tail][relation] = [head]
                    else:
                        self.KG[tail][relation].append(head)
                for type_name in batch_type_to_nodes:
                    self.type_to_nodes[type_name] |= batch_type_to_nodes[type_name]
            
            for head in tqdm(self.KG):
                for relation in self.KG[head]:
                    self.KG[head][relation] = list(set(self.KG[head][relation]))
            for type_name in self.type_to_nodes:
                self.type_to_nodes[type_name] = list(self.type_to_nodes[type_name])

            pickle.dump(self.KG, open(f"{processed_data_dir}/FactKG.graph", "wb"))
            pickle.dump(self.type_to_nodes, open(f"{processed_data_dir}/FactKG.type_to_nodes", "wb"))
        print("- done")
            
        # load queries and groundtruths
        print("Loading queries and groundtruths ...")
        processed_queries_gt_filename = f"{processed_data_dir}/FactKG.queries_gt"
        if os.path.exists(processed_queries_gt_filename):
            self.queries, self.groundtruths = pickle.load(open(processed_queries_gt_filename, 'rb'))
        else:
            test_data = pickle.load(open(f'{raw_data_dir}/factkg_test.pickle', 'rb'))
            self.queries = list(test_data.keys())
            self.groundtruths = []
            for query in self.queries:
                self.groundtruths.append([str(test_data[query]['Label']).replace('[', '').replace(']', '')])
            pickle.dump((self.queries, self.groundtruths), open(processed_queries_gt_filename, 'wb'))
        print("- done")

class ShipQA(Dataset):
    def __init__(self, configs):
        self.hop = configs['hop']
        super().__init__(configs)
        
    def _load_data(self):
        raw_data_dir = self.configs['raw_data_dir']
        processed_data_dir = self.configs['processed_data_dir']
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # load KG
        print("Loading KG ...")
        if os.path.exists(f"{processed_data_dir}/graph"):
            self.KG = pickle.load(open(f"{processed_data_dir}/graph", "rb"))
        else:
            self.KG = {}
            with open(f"{raw_data_dir}/kb.txt", "r", encoding="utf-8") as file:
                for line in file:
                    head, relation, tail = line.strip().split('|')
                    if head not in self.KG:
                        self.KG[head] = {}
                    if tail not in self.KG:
                        self.KG[tail] = {}
                    if relation not in self.KG[head]:
                        self.KG[head][relation] = [tail]
                    else:
                        self.KG[head][relation].append(tail)
                    if relation not in self.KG[tail]:
                        self.KG[tail][relation] = [head]
                    else:
                        self.KG[tail][relation].append(head)
            for head in self.KG:
                for relation in self.KG[head]:
                    self.KG[head][relation] = list(set(self.KG[head][relation]))
            pickle.dump(self.KG, open(f"{processed_data_dir}/graph", "wb"))
        print("- done")
            
        # load queries and groundtruths
        print("Loading queries and groundtruths ...")
        if self.hop == 1:
            qa_filename = f"{raw_data_dir}/1-hop/vanilla/qa_test.txt"
        else:
            qa_filename = f"{raw_data_dir}/{self.hop}-hop/{self.hop}-hop/vanilla/qa_test.txt"
        self.queries, self.groundtruths = [], []
        with open(qa_filename, "r", encoding="utf-8") as file:
            for line in file:
                query, groundtruth = line.strip().split('\t')
                query = query.replace('[', '').replace(']', '')
                groundtruth = groundtruth.split('|')
                self.queries.append(query)
                self.groundtruths.append(groundtruth)
        print("- done")
