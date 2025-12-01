import networkx as nx

file_path='../datasets/background_kbs/pql_3h_kb.txt'

G = nx.DiGraph()
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            h, r, t = parts
            G.add_edge(h, t, relation=r)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")