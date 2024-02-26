from concurrent.futures import ThreadPoolExecutor
import os
from utils.utils import CONSTANTS, dump_jsonl, json_to_graph, CodexTokenizer, load_jsonl, make_needed_dir
import copy
import networkx as nx
import queue
import Levenshtein
import argparse
import time
from utils.metrics import hit
from functools import partial
from build_query_graph import build_query_subgraph


class SimilarityScore:
    @staticmethod
    def text_edit_similarity(str1: str, str2: str):
        return 1 - Levenshtein.distance(str1, str2) / max(len(str1), len(str2))

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

    @staticmethod
    def subgraph_edit_similarity(query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        # To ensure the consistency of sorting scores implementation in the next step, the SED can be straightforwardly transformed into subgraph edit similarity.
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        tokenizer = CodexTokenizer()
        query_graph_node_embedding = tokenizer.tokenize("".join(query_graph.nodes[query_root]['sourceLines']))
        graph_node_embedding = tokenizer.tokenize("".join(graph.nodes[root]['sourceLines']))
        node_sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)

        node_match = dict()
        match_queue = queue.Queue()
        match_queue.put((query_root, root, 0))
        node_match[query_root] = (root, 0)

        query_graph_visited = {query_root}
        graph_visited = {root}

        graph_nodes = set(graph.nodes)

        while not match_queue.empty():
            v, u, hop = match_queue.get()
            v_neighbors = (set(query_graph.neighbors(v)) | set(query_graph.predecessors(v))) - set(query_graph_visited)
            u_neighbors = graph_nodes - set(graph_visited)

            sim_score = []
            for vn in v_neighbors:
                for un in u_neighbors:
                    query_graph_node_embedding = tokenizer.tokenize("".join(query_graph.nodes[vn]['sourceLines']))
                    graph_node_embedding = tokenizer.tokenize("".join(graph.nodes[un]['sourceLines']))
                    sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)
                    sim_score.append((sim, vn, un))
            sim_score.sort(key=lambda x: -x[0])
            for sim, vn, un in sim_score:
                if vn not in query_graph_visited and un not in graph_visited:
                    match_queue.put((vn, un, hop + 1))
                    node_match[vn] = (un, hop + 1)
                    query_graph_visited.add(vn)
                    graph_visited.add(un)
                    v_neighbors.remove(vn)
                    u_neighbors.remove(un)
                    node_sim += (gamma ** (hop + 1)) * sim
                if len(v_neighbors) == 0 or len(u_neighbors) == 0:
                    break
            if len(v_neighbors) != 0:
                for vn in v_neighbors:
                    node_match[vn] = None
                    query_graph_visited.add(vn)

        edge_sim = 0
        for v in query_graph.nodes:
            if v not in node_match.keys():
                node_match[v] = None
        for v_query, u_query, t in query_graph.edges:
            if node_match[v_query] is not None and node_match[u_query] is not None:
                v, hop_v = node_match[v_query]
                u, hop_u = node_match[u_query]
                if graph.has_edge(v, u, t):
                    edge_sim += (gamma ** hop_v)

        graph_sim = node_sim + edge_sim
        return graph_sim


class CodeSearchWorker:
    def __init__(self, query_cases, output_path, mode, gamma=None, max_top_k=CONSTANTS.max_search_top_k, remove_threshold=0):
        self.query_cases = query_cases
        self.output_path = output_path
        self.max_top_k = max_top_k
        self.remove_threshold = remove_threshold
        self.mode = mode
        self.gamma = gamma

    @staticmethod
    def _is_context_after_hole(query_case, repo_case):
        hole_fpath_str = "/".join(query_case['metadata']['fpath_tuple'])
        repo_fpath_str = "/".join(repo_case['fpath_tuple'])
        if hole_fpath_str != repo_fpath_str:
            return False
        else:
            query_case_line = max(query_case['metadata']['forward_context_line_list'])
            repo_case_last_line = repo_case['max_line_no']
            if repo_case_last_line >= query_case_line:
                return True
            else:
                return False

    def _text_jaccard_similarity_wrapper(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0
        sim = SimilarityScore.text_jaccard_similarity(query_case['query_forward_encoding'],
                                                      repo_case['key_forward_encoding'])
        return repo_case, sim

    def _graph_node_prior_similarity_wrapper(self, query_case, repo_case):
        query_graph = json_to_graph(query_case['query_forward_graph'])
        repo_graph = json_to_graph(repo_case['key_forward_graph'])
        if len(repo_graph.nodes) == 0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0
        sim = SimilarityScore.subgraph_edit_similarity(query_graph, repo_graph, gamma=self.gamma)
        return repo_case, sim

    def _find_top_k_context_one_phase(self, query_case):
        start_time = time.time()
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        search_res = copy.deepcopy(query_case)
        repo_cases = load_jsonl(os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl"))
        top_k_context = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            if self.mode == 'coarse':
                compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            else:
                compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            top_k_context = list(futures)
        top_k_context_filtered = []
        for repo_case, sim in top_k_context:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((repo_case['val'], repo_case['statement'],
                                               repo_case['key_forward_context'], repo_case['fpath_tuple'], sim))
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=False)
        search_res['top_k_context'] = top_k_context_filtered[-self.max_top_k:]

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')
        end_time = time.time()
        if self.mode == 'coarse':
            search_res['text_runtime'] = end_time - start_time
            search_res['graph_runtime'] = 0
        else:
            search_res['text_runtime'] = 0
            search_res['graph_runtime'] = end_time - start_time
        return search_res

    def _find_top_k_context_two_phase(self, query_case):
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        repo_cases = load_jsonl(os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl"))
        text_runtime_start = time.time()
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            top_k_context_phase1 = list(futures)
        top_k_context_phase1 = sorted(top_k_context_phase1, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        text_runtime_end = time.time()

        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            top_k_cases = []
            for case, _ in top_k_context_phase1:
                top_k_cases.append(case)
            futures = executor.map(compute_sim, top_k_cases)
            top_k_context_phase2 = list(futures)

        top_k_context_filtered = []
        for repo_case, sim in top_k_context_phase2:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((repo_case['val'], repo_case['statement'],
                                               repo_case['key_forward_context'], repo_case['fpath_tuple'], sim))
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=False)

        query_case['top_k_context'] = top_k_context_filtered[-self.max_top_k:]

        graph_runtime_end = time.time()

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')
        query_case['text_runtime'] = text_runtime_end - text_runtime_start
        query_case['graph_runtime'] = graph_runtime_end - text_runtime_end
        return copy.deepcopy(query_case)

    def run(self):
        query_lines_with_retrieved_results = []
        if self.mode == 'coarse' or self.mode == 'fine':
            for query_case in self.query_cases:
                res = self._find_top_k_context_one_phase(query_case)
                query_lines_with_retrieved_results.append(copy.deepcopy(res))
        else:
            for query_case in self.query_cases:
                res = self._find_top_k_context_two_phase(query_case)
                query_lines_with_retrieved_results.append(copy.deepcopy(res))
        dump_jsonl(query_lines_with_retrieved_results, self.output_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--query_cases', default="api_level", type=str)
    args_parser.add_argument('--mode', type=str, default='coarse2fine')
    args_parser.add_argument('--gamma', default=0.1, type=float)
    args = args_parser.parse_args()

    build_query_subgraph(f"{args.query_cases}.test.jsonl")

    query_cases = load_jsonl(os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl"))

    save_path = os.path.join(f"./search_results/{args.query_cases}.{args.mode}.{args.gamma*100}.search_res.jsonl")
    make_needed_dir(save_path)

    all_start_time = time.time()
    searcher = CodeSearchWorker(query_cases, save_path, args.mode, gamma=args.gamma)
    searcher.run()
    all_end_time = time.time()

    running_time = all_end_time - all_start_time
    search_cases = load_jsonl(save_path)
    hit1, hit5, hit10 = hit(search_cases, hits=[1, 5, 10])

    print('-'*20 + "Parameters" + '-'*20)
    print(f"query_cases: {args.query_cases}")
    print(f'mode: {args.mode}')
    print(f'gamma: {args.gamma}')
    print('-' * 20 + "Results" + '-' * 20)
    print(f'save_path: {save_path}')
    print('hit1 %.4f' % hit1)
    print('hit5 %.4f' % hit5)
    print('hit10 %.4f' % hit10)
    print('runtime %.4f' % running_time)

