import os
import copy
from tqdm import tqdm
import networkx as nx
from utils.ccg import create_graph
from utils.slicing import Slicing
from utils.utils import load_jsonl, make_needed_dir, graph_to_json, CONSTANTS, dump_jsonl, CodexTokenizer


def last_n_context_lines_graph(graph: nx.MultiDiGraph):
    max_line = 0
    last_node_id = 0
    slicer = Slicing()
    for v in graph.nodes:
        if graph.nodes[v]['startRow'] > max_line:
            max_line = graph.nodes[v]['startRow']
            last_node_id = v
    return slicer.forward_dependency_slicing(last_node_id, graph, contain_node=True)


def build_query_subgraph(task_name):
    test_cases = load_jsonl(os.path.join(CONSTANTS.dataset_dir, task_name))
    graph_test_cases = []
    tokenizer = CodexTokenizer()
    with tqdm(total=len(test_cases)) as pbar:
        for case in test_cases:
            # read full query context
            case_path = os.path.join(*case['metadata']['fpath_tuple'])
            line_no = case['metadata']['line_no']
            case_id = case['metadata']['task_id'].split('/')[0]
            if case_id not in CONSTANTS.repos:
                continue
            with open(os.path.join(CONSTANTS.repo_base_dir, case_path), 'r') as f:
                src_lines = f.readlines()
            query_context = src_lines[:line_no]
            ccg = create_graph(query_context, case_id)
            query_ctx, query_line_list, query_graph = last_n_context_lines_graph(ccg)
            graph_case = dict()
            graph_case['query_forward_graph'] = graph_to_json(query_graph)
            graph_case['query_forward_context'] = query_ctx
            graph_case['query_forward_encoding'] = tokenizer.tokenize(query_ctx)
            context_lines = case['prompt'].splitlines(keepends=True)
            graph_case['context'] = context_lines
            graph_case['metadata'] = copy.deepcopy(case['metadata'])
            graph_case['metadata']['forward_context_line_list'] = query_line_list
            graph_test_cases.append(copy.deepcopy(graph_case))
            pbar.update(1)

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, task_name)
    make_needed_dir(save_path)
    dump_jsonl(graph_test_cases, save_path)
    return


if __name__ == "__main__":
    tasks_name = ["api_level.java.test", 
                  "line_level.java.test",
                  "api_level.python.test",
                  "line_level.python.test"]
    for task_name in tasks_name:
        build_query_subgraph(task_name)



