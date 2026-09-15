import argparse
from gnn.dataset_load import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, required=True, help='path to dataset folder, must end with /')
parser.add_argument('--lm', type=str, default='sbert', help='tokenizer: sbert/bert/lstm etc')
parser.add_argument('--is_eval', action='store_true')
args = parser.parse_args()

config = {
    'data_folder': args.data_folder,
    'name': 'custom',
    'max_train': 200000,
    'word2id': 'vocab.txt',
    'relation2id': 'relations.txt',
    'entity2id': 'entities.txt',
    'relation_word_emb': True,
    'is_eval': args.is_eval,
    'q_type': 'seq',
}

dataset = load_data(config, args.lm)
print('Loaded dataset keys:', list(dataset.keys()))
print('Train size:', 0 if dataset['train'] is None else dataset['train'].num_data)
print('Valid size:', dataset['valid'].num_data)
print('Test size:', dataset['test'].num_data)

if dataset['test'].num_data > 0:
    print('\nExample test question:')
    sample = dataset['test'].data[0]
    print(sample.get('question'))
    print('Entities:', sample.get('entities')[:5])
    print('First tuple:', sample.get('subgraph', {}).get('tuples', [])[:5])
