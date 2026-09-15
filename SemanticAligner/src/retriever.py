import time
import math
import copy
import signal
import networkx as nx
from src.utils import kSmallest
from src.vecdb import VectorStore
from src.embedding_model import EmbeddingModel
from concurrent.futures import ThreadPoolExecutor


class Retriever:
	configs = None
	KG = None
	type_to_nodes = None
	use_type_candidates = False
	
	node_vector_store = None
	relation_vector_store = None
	type_vector_store = None
	
	node_sim_topk = None
	relation_sim_topk = None
	type_sim_topk = None
	final_topk = None
	
	def __init__(self, configs, KG, type_to_nodes=None):
		self.configs = configs
		self.KG = KG
		self.model = EmbeddingModel(configs)
		self.timeout = configs['retriever']['timeout']
		self.final_topk = configs['retriever']['final_topk']
		
		self.node_vector_store = VectorStore(self.configs['vector_store_names']['node'])
		assert len(KG) == self.node_vector_store.count(), \
			"The number of nodes in the KG and the node vector store should be the same."
		self.node_sim_topk = self.configs['retriever']['node_sim_topk']     
		
		self.relation_vector_store = VectorStore(self.configs['vector_store_names']['relation'])
		num_relations = len(set([each for node, adj in KG.items() for each in adj.keys()]))
		assert num_relations == self.relation_vector_store.count(), \
			"The number of relations in the KG and the relation vector store should be the same."
		self.relation_sim_topk = self.configs['retriever']['relation_sim_topk']
			
		if type_to_nodes:
			self.use_type_candidates = True
			self.type_to_nodes = type_to_nodes
			self.type_vector_store = VectorStore(self.configs['vector_store_names']['type'])
			assert len(type_to_nodes) == self.type_vector_store.count(), \
				"The number of types in the type_to_nodes and the type vector store should be the same."
			self.type_sim_topk = self.configs['retriever']['type_sim_topk']
			
	def _dfs_all_edges(self, Q, node, visited_edges):
		neighbors_with_degrees = [(neighbor, Q.degree(neighbor)) for neighbor in Q[node]]
		for neighbor, _ in sorted(neighbors_with_degrees, key=lambda x: x[1], reverse=True):
			if (node, neighbor) not in visited_edges and (neighbor, node) not in visited_edges:
				visited_edges.append((node, neighbor))
				visited_edges = self._dfs_all_edges(Q, neighbor, visited_edges)
			if len(visited_edges) == Q.number_of_edges():
				return visited_edges
		return visited_edges     
			
	def retrieve(self, query_graph, mode='greedy'):
		
  		# transform into networkx graph
		Q = nx.Graph()
		for edge in query_graph:
			Q.add_edge(edge[0], edge[2], relation=edge[1], matched=False)
   
		# sub queries for disconnected components
		if not nx.is_connected(Q):
			result = {
				"embedding_time": 0,
				"vector_search_time": 0,
				"graph_search_time": 0,
				"results": []
			}
			for component in nx.connected_components(Q):
				subQ_edges = [each for each in query_graph if each[0] in component and each[2] in component]
				res = self.retrieve(subQ_edges, mode)
				result['embedding_time'] += res['embedding_time']
				result['vector_search_time'] += res['vector_search_time']
				result['graph_search_time'] += res['graph_search_time']
				result["results"].extend(res["results"])
			return result

		# set time out
		signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(Exception("retrieval run time out")))
		signal.alarm(self.timeout)
	 
		try:
			# extract the query nodes and relations
			query_nodes = [node for node in Q.nodes() if "UNKNOWN" not in node]
			unknown_nodes = [node for node in Q.nodes() if "UNKNOWN" in node]
			query_relations = list(set([data['relation'] for u, v, data in Q.edges(data=True) if "UNKNOWN" not in data['relation']]))
			
			# encode the query nodes and relations
			start = time.time()
			if self.use_type_candidates and len(query_nodes) == 0:
				query_texts = query_nodes + [each.replace("UNKNOWN", "") for each in unknown_nodes] + query_relations
				query_embeddings = self.model.encode(query_texts)
				query_node_vectors = query_embeddings[:len(query_nodes)]
				query_type_vectors = query_embeddings[len(query_nodes):len(query_nodes)+len(unknown_nodes)]
				query_relation_vectors = query_embeddings[len(query_nodes)+len(unknown_nodes):]
			else:
				query_texts = query_nodes + query_relations
				query_embeddings = self.model.encode(query_texts)
				query_node_vectors = query_embeddings[:len(query_nodes)]
				query_type_vectors = []
				query_relation_vectors = query_embeddings[len(query_nodes):]
			embedding_time = time.time() - start
		
			# search for similar nodes and relations
			start = time.time()
			def search_node_vectors():
				return {
					query_node: {each['entity']['name']: math.sqrt(each['distance']) for each in result if each['entity']['name'] in self.KG}
					for query_node, result in zip(query_nodes, 
												self.node_vector_store.search(query_node_vectors, self.node_sim_topk))
				} if query_nodes else {}
			def search_relation_vectors():
				similar_relations = {
										query_relation: {each['entity']['name']: math.sqrt(each['distance']) for each in result}
										for query_relation, result in zip(query_relations, 
																		self.relation_vector_store.search(query_relation_vectors, self.relation_sim_topk))
									} if query_relations else {}
				for u, v, data in Q.edges(data=True):
					if "UNKNOWN" in data['relation']:
						similar_relations[data['relation']] = None
				return similar_relations
			def search_type_vectors():
				similar_nodes = {}
				if len(query_type_vectors) > 0:
					query_type_vector_search_results = self.type_vector_store.search(query_type_vectors, self.type_sim_topk) 
					for node, query_type_vector_search_result in zip(unknown_nodes, query_type_vector_search_results):
						similar_nodes[node] = {}
						for each in query_type_vector_search_result:
							type_name = each['entity']['name']
							for KG_node in self.type_to_nodes[type_name]:
								if KG_node in self.KG:
									similar_nodes[node][KG_node] = 0
				else:
					similar_nodes = {node: None for node in unknown_nodes}
				return similar_nodes
			with ThreadPoolExecutor() as executor:
				futures = {
					'node_vectors': executor.submit(search_node_vectors),
					'relation_vectors': executor.submit(search_relation_vectors),
					'type_vectors': executor.submit(search_type_vectors)
				}
			results = {key: future.result() for key, future in futures.items()}
			similar_nodes = {**results['node_vectors'], **results['type_vectors']}
			similar_relations = results['relation_vectors']
			vector_search_time = time.time() - start
		
			# DFS matching order
			root = sorted(similar_nodes.keys(), key=lambda x: (len(similar_nodes[x]) if similar_nodes[x] else len(self.KG)))[0]
			if not similar_nodes[root]:
				raise Exception("No similar nodes found for the root, too many unknown nodes")
			Q_edge_sequence = self._dfs_all_edges(Q, root, [])

			# for greedy search pruning
			if mode == 'greedy':
				final_best_score = [0] * Q.number_of_edges()
				for i in range(Q.number_of_edges()-1, 0, -1):
					cur_Q_node, Q_neighbor = Q_edge_sequence[i]
					Q_relation = Q[cur_Q_node][Q_neighbor]['relation']
					final_best_score[i-1] = final_best_score[i]
					if similar_relations[Q_relation]:
						final_best_score[i-1] += min(similar_relations[Q_relation].values())
					if similar_nodes[Q_neighbor]:
						final_best_score[i-1] += min(similar_nodes[Q_neighbor].values())
			
			# DFS matching
			def match(cur_Q_idx, node_matching, matched_KG_edges, cur_score, reuse_nodes):
				if cur_Q_idx == Q.number_of_edges():
					results.add(cur_score, copy.deepcopy(matched_KG_edges), reuse_nodes)
					return
				
				cur_Q_node, Q_neighbor = Q_edge_sequence[cur_Q_idx]
				Q_relation = Q[cur_Q_node][Q_neighbor]['relation']
				cur_KG_node = node_matching[cur_Q_node]

				to_expand = []
				for KG_relation in self.KG[cur_KG_node]:
					if similar_relations[Q_relation] is None or KG_relation in similar_relations[Q_relation]:
						for KG_neighbor in self.KG[cur_KG_node][KG_relation]:
							if similar_nodes[Q_neighbor] is None or KG_neighbor in similar_nodes[Q_neighbor]:
								next_reuse_nodes = reuse_nodes
								if Q_neighbor in node_matching:
									if node_matching[Q_neighbor] != KG_neighbor:
										continue	# not consistent
									node_score = 0
								else:
									if KG_neighbor in node_matching.values():
										next_reuse_nodes = True
									node_score = 0 if similar_nodes[Q_neighbor] is None else similar_nodes[Q_neighbor][KG_neighbor]
								relation_score = 0 if similar_relations[Q_relation] is None else similar_relations[Q_relation][KG_relation]
								to_expand.append((KG_relation, KG_neighbor, cur_score + node_score + relation_score, next_reuse_nodes))
								
				# greedy expansion
				if mode == 'greedy':
					to_expand = sorted(to_expand, key=lambda x: x[2])
					for KG_relation, KG_neighbor, next_score, next_reuse_nodes in to_expand:
						if next_score + final_best_score[cur_Q_idx] > results.max_score():
							break
						
						next_node_matching = copy.deepcopy(node_matching)
						next_node_matching[Q_neighbor] = KG_neighbor
						next_matched_KG_edges = matched_KG_edges + [(cur_KG_node, KG_relation, KG_neighbor)]
						match(cur_Q_idx + 1, next_node_matching, next_matched_KG_edges, next_score, next_reuse_nodes)
				
				# naive search
				else:            
					for KG_relation, KG_neighbor, next_score, next_reuse_nodes in to_expand:   
						next_node_matching = copy.deepcopy(node_matching)
						next_node_matching[Q_neighbor] = KG_neighbor
						next_matched_KG_edges = matched_KG_edges + [(cur_KG_node, KG_relation, KG_neighbor)]
						match(cur_Q_idx + 1, next_node_matching, next_matched_KG_edges, next_score, next_reuse_nodes)
				
			# graph search
			start = time.time()
			results = kSmallest(self.final_topk)
			for KG_node, distance in sorted(similar_nodes[root].items(), key=lambda x: x[1]):
				match(0, {root: KG_node}, [], distance, False)
			graph_search_time = time.time() - start
			
			final_results = results.get()
			# 如果图搜索没有任何完整匹配，回退到基于向量检索的候选（node-level），以避免返回空结果
			if not final_results:
				fallback = []
				seen = set()
				# 从 similar_nodes 收集候选（按距离升序）
				for qnode, candidates in similar_nodes.items():
					if not candidates:
						continue
					for cand_name, dist in sorted(candidates.items(), key=lambda x: x[1]):
						if cand_name in seen:
							continue
						seen.add(cand_name)
						# 构造单跳证据：把匹配节点放入一个单边三元组，relation 使用特殊标记
						fallback.append((dist, [(cand_name, '[approx_match]', cand_name)], False))
						if len(fallback) >= self.final_topk:
							break
					if len(fallback) >= self.final_topk:
						break
				final_results = fallback
			
			return {
				"embedding_time": embedding_time,
				"vector_search_time": vector_search_time,
				"graph_search_time": graph_search_time,
				"results": final_results
			}
		except Exception as e:
			raise e
		finally:
			signal.alarm(0)