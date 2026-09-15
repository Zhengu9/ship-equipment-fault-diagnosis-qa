
import torch.nn.functional as F
import torch.nn as nn
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


from transformers import AutoModel, AutoTokenizer #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.nn import LayerNorm
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'

from .base_encoder import BaseInstruction


class BERTInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        
        if model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.pretrained_weights = 'bert-base-chinese'
            word_dim = 768#self.word_dim
        elif model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            self.pretrained_weights = 'roberta-base'
            word_dim = 768#self.word_dim
        elif model == 'sbert':
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            local_sbert = os.path.join(repo_root, 'pretrained_lms', 'all-MiniLM-L6-v2')
            local_sr = os.path.join(repo_root, 'pretrained_lms', 'sr-simbert')
            if os.path.isdir(local_sbert):
                self.tokenizer = AutoTokenizer.from_pretrained(local_sbert)
                self.pretrained_weights = local_sbert
            elif os.path.isdir(local_sr):
                self.tokenizer = AutoTokenizer.from_pretrained(local_sr)
                self.pretrained_weights = local_sr
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                self.pretrained_weights = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            word_dim = 384#self.word_dim
        elif model == 'simcse':
            #print('ok')
            self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
            self.pretrained_weights = 'princeton-nlp/sup-simcse-bert-base-uncased'
            word_dim = 768#self.word_dim
        elif model == 'sbert2':
            #tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            local_mpnet = os.path.join(repo_root, 'pretrained_lms', 'all-mpnet-base-v2')
            if os.path.isdir(local_mpnet):
                self.tokenizer = AutoTokenizer.from_pretrained(local_mpnet)
                self.pretrained_weights = local_mpnet
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
                self.pretrained_weights = 'sentence-transformers/all-mpnet-base-v2'
            word_dim = 768#self.word_dim
        elif model == 't5':
            self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
            self.pretrained_weights = 't5-small'
            word_dim = 768#self.word_dim
        elif model  == 'relbert':
            self.tokenizer = AutoTokenizer.from_pretrained('pretrained_lms/sr-simbert/')
            self.pretrained_weights = 'pretrained_lms/sr-simbert/'
            word_dim = 768
        elif model == 'xlm':
            # prefer a local xlm-roberta-base under repo root if present
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            local_xlm = os.path.join(repo_root, 'xlm-roberta-base')
            try:
                if os.path.isdir(local_xlm):
                    self.tokenizer = AutoTokenizer.from_pretrained(local_xlm)
                    self.pretrained_weights = local_xlm
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                    self.pretrained_weights = 'xlm-roberta-base'
            except Exception:
                # fallback to local files only if needed
                self.tokenizer = AutoTokenizer.from_pretrained(local_xlm, local_files_only=True)
                self.pretrained_weights = local_xlm
            word_dim = 768
        #self.mask = mask
        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.word_dim = word_dim

        print('word_dim', self.word_dim)
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        # question_emb depends on the LM hidden size; define it in encoder_def
        self.question_emb = None

        if not self.constraint:
            self.encoder_def()

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        entity_dim = self.entity_dim
        self.node_encoder = AutoModel.from_pretrained(self.pretrained_weights)
        # set word_dim from the actual LM hidden size if available
        try:
            lm_hidden = getattr(self.node_encoder.config, 'hidden_size', None)
            if lm_hidden is not None:
                self.word_dim = lm_hidden
        except Exception:
            pass
        # (re)create question_emb with correct in_features
        self.question_emb = nn.Linear(in_features=self.word_dim, out_features=entity_dim)
        print('Total Params', sum(p.numel() for p in self.node_encoder.parameters()))
        if self.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.node_encoder.parameters():
                param.requires_grad = True
            print('Unfrozen LM params')

    def encode_question(self, query_text, store=True):
        batch_size = query_text.size(0)
        
        if self.model != 't5':
            
            query_hidden_emb = self.node_encoder(query_text)[0]  # 1, batch_size, entity_dim
        else:
            query_hidden_emb = self.node_encoder.encoder(query_text)[0]
            #print(query_hidden_emb.size())
        

        if store:
            # apply linear to last dim safely for both 2D and 3D tensors
            if query_hidden_emb.dim() == 3:
                b, s, h = query_hidden_emb.size()
                self.query_hidden_emb = self.question_emb(query_hidden_emb.contiguous().view(-1, h)).view(b, s, -1)
            else:
                self.query_hidden_emb = self.question_emb(query_hidden_emb)

            self.query_node_emb = query_hidden_emb.transpose(1,0)[0].unsqueeze(1)
            # apply linear to last dim of node embedding as well
            if self.query_node_emb.dim() == 3:
                nb, ns, nh = self.query_node_emb.size()
                self.query_node_emb = self.question_emb(self.query_node_emb.contiguous().view(-1, nh)).view(nb, ns, -1)
            else:
                self.query_node_emb = self.question_emb(self.query_node_emb)
            
            self.query_mask = (query_text != self.pad_val).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return  query_hidden_emb 

