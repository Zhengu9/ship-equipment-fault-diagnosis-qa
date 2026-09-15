import os
import pickle
from tqdm import tqdm
from src.vecdb import VectorStore
from src.embedding_model import EmbeddingModel


class Indexer:
	def __init__(self, configs):
		self.configs = configs
		self.model = EmbeddingModel(configs)

	def build_index(self, KG, type_to_nodes=None):
		processed_data_dir = self.configs['processed_data_dir']
		
		# generate node embddings in batchs
		print("Generating node embeddings ...")
		batch_size = 8192
		nodes = sorted(list(KG.keys()))
		node_name_to_id = {}
		os.makedirs(f"{processed_data_dir}/node_embeddings", exist_ok=True)
		os.makedirs(f"{processed_data_dir}/node_name_to_id", exist_ok=True)
		for batch_start in tqdm(range(0, len(nodes), batch_size)):
			if os.path.exists(f"{processed_data_dir}/node_embeddings/{batch_start // batch_size}") \
			   and os.path.exists(f"{processed_data_dir}/node_name_to_id/{batch_start // batch_size}"):
				continue
			batch_nodes = nodes[batch_start:(batch_start + batch_size)]
			batch_embeddings = [{
				'id': batch_start + i,
				'name': node,
				'vector': embedding.tolist()
			} for i, (node, embedding) in enumerate(zip(batch_nodes, self.model.encode(batch_nodes)))]
			node_name_to_id = ({node['name']: node['id'] for node in batch_embeddings})
			pickle.dump(batch_embeddings, open(f"{processed_data_dir}/node_embeddings/{batch_start // batch_size}", "wb"))
			pickle.dump(node_name_to_id, open(f"{processed_data_dir}/node_name_to_id/{batch_start // batch_size}", "wb"))
		print("- done")

		# generate relation embeddings
		print("Generating relation embeddings ...")
		relations = list(set([each for node, adj in KG.items() for each in adj.keys()]))
		if (not os.path.exists(f"{processed_data_dir}/relation_embeddings")) \
			or (not os.path.exists(f"{processed_data_dir}/relation_name_to_id")):
			relation_embeddings = [{
				'id': i,
				'name': relation,
				'vector': embedding.tolist()
			} for i, (relation, embedding) in enumerate(zip(relations, self.model.encode(relations, show_progress_bar=True)))]
			relation_name_to_id = ({relation['name']: relation['id'] for relation in relation_embeddings})
			pickle.dump(relation_embeddings, open(f"{processed_data_dir}/relation_embeddings", "wb"))
			pickle.dump(relation_name_to_id, open(f"{processed_data_dir}/relation_name_to_id", "wb"))
		print("- done")

		# generate type embeddings
		if type_to_nodes:
			print("Generating type embeddings ...")
			types = list(type_to_nodes.keys())
			if (not os.path.exists(f"{processed_data_dir}/type_embeddings")) \
				or (not os.path.exists(f"{processed_data_dir}/type_name_to_id")):
				type_embeddings = [{
					'id': i,
					'name': type,
					'vector': embedding.tolist()
				} for i, (type, embedding) in enumerate(zip(types, self.model.encode(types, show_progress_bar=True)))]
				type_name_to_id = ({type['name']: type['id'] for type in type_embeddings})
				pickle.dump(type_embeddings, open(f"{processed_data_dir}/type_embeddings", "wb"))
				pickle.dump(type_name_to_id, open(f"{processed_data_dir}/type_name_to_id", "wb"))
			print("- done")
				
		# write node embeddings to vector store
		print("Writing node embeddings to vector store ...")
		node_vector_store = VectorStore(self.configs["vector_store_names"]["node"])
		if node_vector_store.count() != len(nodes):
			if node_vector_store.count() > 0:
				node_vector_store.reset()
			for batch_start in tqdm(range(0, len(nodes), batch_size)):
				batch_embeddings = pickle.load(open(f"{processed_data_dir}/node_embeddings/{batch_start // batch_size}", "rb"))
				node_vector_store.insert(data=batch_embeddings)
		print("- done")

		# write relation embeddings to vector store
		print("Writing relation embeddings to vector store ...")
		relation_vector_store = VectorStore(self.configs["vector_store_names"]["relation"])
		if relation_vector_store.count() != len(relations):
			if relation_vector_store.count() > 0:
				relation_vector_store.reset()
			relation_embeddings = pickle.load(open(f"{processed_data_dir}/relation_embeddings", "rb"))
			for i in tqdm(range(0, len(relation_embeddings), batch_size)):
				relation_vector_store.insert(data=relation_embeddings[i:i+batch_size])
		print("- done")
		
		# write type embeddings to vector store
		if type_to_nodes:
			print("Writing type embeddings to vector store ...")
			type_vector_store = VectorStore(self.configs["vector_store_names"]["type"])
			if type_vector_store.count() != len(types):
				if type_vector_store.count() > 0:
					type_vector_store.reset()
				type_embeddings = pickle.load(open(f"{processed_data_dir}/type_embeddings", "rb"))
				for i in tqdm(range(0, len(type_embeddings), batch_size)):
					type_vector_store.insert(data=type_embeddings[i:i+batch_size])
			print("- done")