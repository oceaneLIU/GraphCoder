import networkx as nx
from tree_sitter import Language, Parser


def python_control_dependence_graph(root_node, CCG, src_lines, parent):
    node_id = len(CCG.nodes)

    if root_node.type in ['import_from_statement', 'import_statement']:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]

        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ['class_definition', 'decorated_definition', 'function_definition']:
        if root_node.type == 'function_definition':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('parameters').end_point[0]
        elif root_node.type == 'decorated_definition':
            def_node = root_node.child_by_field_name('definition')
            start_row = root_node.start_point[0]
            parameter_node = def_node.child_by_field_name('parameters')
            if parameter_node is not None:
                end_row = parameter_node.end_point[0]
            else:
                end_row = def_node.start_point[0]
        elif root_node.type == 'class_definition':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('name').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ['while_statement', 'for_statement']:
        if root_node.type == 'for_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('right').end_point[0]
        if root_node.type == 'while_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type == 'if_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type == 'elif_clause':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ['else_clause', 'except_clause']:
        start_row = root_node.start_point[0]
        end_row = root_node.start_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type == 'with_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.children[1].end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif 'statement' in root_node.type or 'ERROR' in root_node.type:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    for child in root_node.children:
        if child.type == 'identifier':
            row = child.start_point[0]
            col_start = child.start_point[1]
            col_end = child.end_point[1]
            identifier_name = src_lines[row][col_start:col_end].strip()
            if parent is None:
                continue
            if 'definition' in CCG.nodes[parent]['nodeType']:
                CCG.nodes[parent]['defSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'for_statement':
                p = child
                while p.parent.type != 'for_statement':
                    p = p.parent
                if p.parent.type == 'for_statement' and p.prev_sibling.type == 'for':
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'with_statement':
                if child.parent.type == 'as_pattern_target':
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] == 'expression_statement':
                p = child
                while p.parent.type != 'assignment' and p.parent.type != 'expression_statement':
                    p = p.parent
                if p.parent.type == 'assignment' and p.next_sibling is not None:
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif 'import' in CCG.nodes[parent]['nodeType']:
                CCG.nodes[parent]['defSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] in ['global_statement', 'nonlocal_statement']:
                CCG.nodes[parent]['defSet'].add(identifier_name)
            else:
                CCG.nodes[parent]['useSet'].add(identifier_name)
        python_control_dependence_graph(child, CCG, src_lines, parent)

    return


def python_control_flow_graph(CCG):
    CFG = nx.MultiDiGraph()

    next_sibling = dict()
    first_children = dict()

    start_nodes = []
    for v in CCG.nodes:
        if len(list(CCG.predecessors(v))) == 0:
            start_nodes.append(v)
    start_nodes.sort()
    for i in range(0, len(start_nodes) - 1):
        v = start_nodes[i]
        u = start_nodes[i + 1]
        next_sibling[v] = u
    next_sibling[start_nodes[-1]] = None

    for v in CCG.nodes:
        children = list(CCG.neighbors(v))
        if len(children) != 0:
            children.sort()
            for i in range(0, len(children) - 1):
                u = children[i]
                w = children[i + 1]
                if CCG.nodes[v]['nodeType'] == 'if_statement' and 'clause' in CCG.nodes[w]['nodeType']:
                    next_sibling[u] = None
                else:
                    next_sibling[u] = w
            next_sibling[children[-1]] = None
            first_children[v] = children[0]
        else:
            first_children[v] = None

    edge_list = []

    for v in CCG.nodes:
        # block start control flow
        if v in first_children.keys():
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        # block end control flow
        if CCG.nodes[v]['nodeType'] in ['return_statement', 'raise_statement']:
            pass
        elif CCG.nodes[v]['nodeType'] in ['break_statement', 'continue_statement']:
            u = None
            p = list(CCG.predecessors(v))[0]
            while CCG.nodes[p]['nodeType'] not in ['for_statement', 'while_statement']:
                p = list(CCG.predecessors(p))[0]
            if CCG.nodes[v]['nodeType'] == 'break_statement':
                u = next_sibling[p]
            if CCG.nodes[v]['nodeType'] == 'continue_statement':
                u = p
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] == 'for_statement':
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] == 'while_statement':
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            u = next_sibling[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] in ['if_statement' or 'try_statement']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            for u in CCG.neighbors(v):
                if 'clause' in CCG.nodes[u]['nodeType']:
                    edge_list.append((v, u, 'CFG'))
        elif 'clause' in CCG.nodes[v]['nodeType']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))

        u = next_sibling[v]
        if u is None:
            p = v
            while len(list(CCG.predecessors(p))) != 0:
                p = list(CCG.predecessors(p))[0]
                if CCG.nodes[p]['nodeType'] == 'while_statement':
                        edge_list.append((v, p, 'CFG'))
                        break
                if CCG.nodes[p]['nodeType'] == 'for_statement':
                    edge_list.append((v, p, 'CFG'))
                    break
                if CCG.nodes[p]['nodeType'] in ['try_statement', 'if_statement']:
                    if next_sibling[p] is not None:
                        edge_list.append((v, next_sibling[p], 'CFG'))
                        break
        if u is not None:
            edge_list.append((v, u, 'CFG'))
    CFG.add_edges_from(edge_list)
    for v in CCG.nodes:
        if v not in CFG.nodes:
            CFG.add_node(v)
    return CFG, edge_list


def python_data_dependence_graph(CFG, CCG):
    for v in CCG.nodes:
        for u in CCG.nodes:
            if v == u or 'import' in CCG.nodes[v]['nodeType']:
                continue
            # find the definition of u
            u_def = u
            u_def_set = set()
            while len(list(CCG.predecessors(u_def))) != 0:
                u_def = list(CCG.predecessors(u_def))[0]
                if 'definition' in CCG.nodes[u_def]['nodeType']:
                    u_def_set.add(u_def)
            if 'definition' in CCG.nodes[v]['nodeType'] and v not in u_def_set:
                continue
            if len(CCG.nodes[v]['defSet'] & CCG.nodes[u]['useSet']) != 0 and nx.has_path(CFG, v, u):
                has_path = False
                paths = list(nx.all_shortest_paths(CFG, source=v, target=u))
                variables = CCG.nodes[v]['defSet'] & CCG.nodes[u]['useSet']
                for var in variables:
                    has_def = False
                    for path in paths:
                        for p in path[1:-1]:
                            if var in CCG.nodes[p]['defSet']:
                                has_def = True
                                break
                        if not has_def:
                            has_path = True
                            break
                    if has_path:
                        break
                if has_path:
                    CCG.add_edge(v, u, 'DDG')
    return


def create_python_graph(code_lines):

    src_lines = "".join(code_lines).encode('ascii', errors='ignore').decode('ascii')
    src_lines = src_lines.splitlines(keepends=True)

    if len(src_lines) != 0:
        src_lines[-1] = src_lines[-1].rstrip().strip('(').strip('[').strip(',')
    # Define tree-sitter python parser
    Language.build_library('./my-languages.so', ['./tree-sitter-python'])
    py_language = Language('./my-languages.so', 'python')
    parser = Parser()
    parser.set_language(py_language)

    if len(src_lines) == 0:
        return None
    # remove comment
    comment_lines = []
    for i in range(0, len(src_lines)):
        line = src_lines[i]
        if line.lstrip().startswith('#'):
            src_lines[i] = '\n'
            comment_lines.append(i)

    # Parser python file to get a tree
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(src_lines) or column >= len(src_lines[row]):
            return None
        return src_lines[row][column:].encode('utf8', errors='ignore')
    tree = parser.parse(read_callable)

    all_comment = True
    for child in tree.root_node.children:
        if child.type != 'comment':
            all_comment = False
    if all_comment:
        return None

    # Initialize program dependence graph
    ccg = nx.MultiDiGraph()

    # Construct control dependence edge
    for child in tree.root_node.children:
        python_control_dependence_graph(child, ccg, code_lines, None)

    # Construct control flow graph
    cfg, cfg_edge_list = python_control_flow_graph(ccg)

    # Construct data dependence graph
    python_data_dependence_graph(cfg, ccg)

    ccg.add_edges_from(cfg_edge_list)

    # add comment
    node_list = list(ccg.nodes)
    node_list.sort()
    comment_lines.reverse()
    max_comment_line = 0
    for comment_line_num in comment_lines:
        insert_id = -1
        for v in ccg.nodes:
            if ccg.nodes[v]['startRow'] > comment_line_num:
                insert_id = v
                break
        if insert_id == -1:
            max_comment_line = max(max_comment_line, comment_line_num)
        else:
            ccg.nodes[insert_id]['startRow'] = comment_line_num
            end_row = ccg.nodes[insert_id]['endRow']
            ccg.nodes[insert_id]['sourceLines'] = code_lines[comment_line_num: end_row + 1]
    if max_comment_line != 0:
        last_node_id = node_list[-1]
        ccg.nodes[last_node_id]['endRow'] = max_comment_line
        start_row = ccg.nodes[last_node_id]['startRow']
        ccg.nodes[last_node_id]['sourceLines'] = code_lines[start_row: max_comment_line + 1]
    return ccg
