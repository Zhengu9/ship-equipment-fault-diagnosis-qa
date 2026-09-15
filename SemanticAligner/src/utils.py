import re
import heapq
import json
import ast


def _normalize_text(s: str) -> str:
	# normalize common full-width / Chinese punctuation and quotes to ascii
	replace_map = {
		'“': '"', '”': '"', '‘': "'", '’': "'",
		'，': ',', '：': ':', '；': ';', '。': '.', '、': ',',
		'【': '[', '】': ']', '（': '(', '）': ')', '《': '"', '》': '"'
	}
	for k, v in replace_map.items():
		s = s.replace(k, v)
	# remove BOM if present
	if s.startswith('\ufeff'):
		s = s.lstrip('\ufeff')
	return s


def extract_graph(llm_output):
	s = _normalize_text(llm_output or '')

	# 1) Try to find a JSON-like object and parse it
	# Attempt to find any balanced JSON-like objects and parse each one.
	# This handles cases where the LLM emits multiple objects or extra text between them.
	objs = []
	stack = []
	for i, ch in enumerate(s):
		if ch == '{':
			stack.append(i)
		elif ch == '}' and stack:
			start = stack.pop()
			end = i
			objs.append((start, end))
	# try parsing each candidate object (from left to right)
	for start, end in objs:
		candidate = s[start:end+1]
		try:
			decoded = json.loads(candidate)
			if isinstance(decoded, dict) and 'graph' in decoded:
				return decoded['graph']
		except Exception:
			try:
				decoded = ast.literal_eval(candidate)
				if isinstance(decoded, dict) and 'graph' in decoded:
					return decoded['graph']
			except Exception:
				pass

	# 2) Try to find a "graph": [...] pattern and parse the list
	m = re.search(r'"graph"\s*:\s*(\[[\s\S]*\])', s)
	if m:
		candidate = m.group(1)
		try:
			decoded = json.loads(candidate)
			return decoded
		except Exception:
			try:
				decoded = ast.literal_eval(candidate)
				return decoded
			except Exception:
				pass

	# If the simple regex fails (LLM output may contain extra text/comments),
	# try a manual bracket-matching extraction after the "graph" token.
	idx = s.find('"graph"')
	if idx != -1:
		# find the colon after "graph"
		colon = s.find(':', idx)
		if colon != -1:
			# find the first '[' after the colon
			start = s.find('[', colon)
			if start != -1:
				depth = 0
				for i in range(start, len(s)):
					ch = s[i]
					if ch == '[':
						depth += 1
					elif ch == ']':
						depth -= 1
						if depth == 0:
							candidate = s[start:i+1]
							try:
								decoded = json.loads(candidate)
								return decoded
							except Exception:
								try:
									decoded = ast.literal_eval(candidate)
									return decoded
								except Exception:
									pass

	# 3) As a last resort, extract list-like triples with regex
	triples = re.findall(r"\[\s*['\"]?([^'\"]+?)['\"]?\s*,\s*['\"]?([^'\"]+?)['\"]?\s*,\s*['\"]?([^'\"]+?)['\"]?\s*\]", s)
	if triples:
		return [[a.strip(), b.strip(), c.strip()] for a, b, c in triples]

	# nothing found — include a snippet of the LLM output in the error to aid debugging
	snippet = s.strip()[:500]
	raise Exception(f"Fail to decode rewrited graph; sample output: {snippet}")


class kSmallest:
	def __init__(self, k):
		self.k = k
		self.heap = []

	def add(self, score, graph, reuse_nodes):
		heapq.heappush(self.heap, (-score, graph, reuse_nodes))
		if len(self.heap) > self.k:
			all_scores = sorted([-neg_score for neg_score, _, _ in self.heap])
			if all_scores[-1] > all_scores[self.k-1]:
				heapq.heappop(self.heap)
			
	def get(self):
		data = sorted([(-neg_score, graph, reuse_nodes) for neg_score, graph, reuse_nodes in self.heap])
  
		final_results = []
		for score, graph, reuse_nodes in data:
			if not reuse_nodes:
				final_results.append((score, graph, reuse_nodes))
				data.remove((score, graph, reuse_nodes))
				break
		for score, graph, reuse_nodes in data:
			if reuse_nodes:
				final_results.append((score, graph, reuse_nodes))
				data.remove((score, graph, reuse_nodes))
				break
   
		for score, graph, reuse_nodes in data:
			if len(final_results) >= self.k:
				break
			final_results.append((score, graph, reuse_nodes))
		return final_results

	def max_score(self):
		if len(self.heap) < self.k:
			return float('inf')
		return -min(self.heap)[0]


def check_answer(answer, groundtruths):
	for gt in groundtruths:
		if gt.lower() in answer.lower():
			return True
	return False