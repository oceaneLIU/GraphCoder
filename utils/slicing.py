import queue
import networkx as nx
from utils.utils import CONSTANTS


class Slicing:

    def __init__(self, max_hop=CONSTANTS.max_hop, max_statement=CONSTANTS.max_statement):
        self.max_hop = max_hop
        self.max_statement = max_statement

    def forward_dependency_slicing(self, node, graph: nx.MultiDiGraph, contain_node=False):
        line_ctx = dict()
        visited = set()
        n_nodes = len(graph.nodes)

        q = queue.Queue()
        q.put((node, 0))

        def cdg_view(v, u, t):
            return t == 'CDG'

        def cfg_view(v, u, t):
            return t == 'CFG'

        def ddg_view(v, u, t):
            return t == 'DDG'

        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        cfg = nx.subgraph_view(graph, filter_edge=cfg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)

        n_statement = set()
        while len(n_statement) < self.max_statement:
            if len(n_statement) == n_nodes or q.empty():
                break
            curr_v, hop = q.get()
            start_line = graph.nodes[curr_v]['startRow']
            end_line = graph.nodes[curr_v]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx[i] = graph.nodes[curr_v]['sourceLines'][i - start_line]
            n_statement.add(curr_v)
            if len(n_statement) >= self.max_statement:
                break
            p = curr_v
            if p in cdg.nodes:
                while len(list(cdg.predecessors(p))) != 0:
                    p = list(cdg.predecessors(p))[0]
                    start_line = graph.nodes[p]['startRow']
                    end_line = graph.nodes[p]['endRow']
                    for i in range(start_line, end_line + 1):
                        line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                    n_statement.add(p)
                    if len(n_statement) >= self.max_statement:
                        break

            for u in ddg.predecessors(curr_v):
                p = u
                start_line = graph.nodes[p]['startRow']
                end_line = graph.nodes[p]['endRow']
                for i in range(start_line, end_line + 1):
                    line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                n_statement.add(p)
                if len(n_statement) >= self.max_statement:
                    break
                if p in cdg.nodes:
                    if len(list(cdg.predecessors(p))) != 0:
                        p = list(cdg.predecessors(p))[0]
                        start_line = graph.nodes[p]['startRow']
                        end_line = graph.nodes[p]['endRow']
                        for i in range(start_line, end_line + 1):
                            line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                        n_statement.add(p)

            if hop+1 > self.max_hop:
                continue
            else:
                for u in cfg.predecessors(curr_v):
                    if u not in visited and 'definition' not in graph.nodes[u]['nodeType']:
                        q.put((u, hop+1))
            visited.add(curr_v)

        if not contain_node:
            n_statement.remove(node)
            start_line = graph.nodes[node]['startRow']
            end_line = graph.nodes[node]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx.pop(i)

        line_list = list(line_ctx.keys())
        line_list.sort()
        ctx = []
        for i in range(0, len(line_list)):
            ctx.append(line_ctx[line_list[i]])
        subgraph = nx.subgraph(graph, n_statement)

        return "".join(ctx), line_list, subgraph

