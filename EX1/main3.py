import networkx as nx
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix

def load_graph(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                src, dst = map(int, line.strip().split())
                G.add_edge(src, dst)
    return G

def pagerank_step(args):
    d, M_T, N, start, end, pr_old = args
    pr_new = d * M_T[start:end].dot(pr_old) + (1 - d) / N
    return pr_new

def calculate_pagerank_parallel(graph, d=0.85, max_iter=100, tol=1e-4, processes=4):
    nodes = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)

    row, col = [], []
    for src, dst in graph.edges():
        row.append(node_to_index[dst])
        col.append(node_to_index[src])
    data = np.ones(len(row), dtype=np.float32)
    M = csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float32)

    out_degree = np.array(M.sum(axis=1)).flatten()
    for i in range(len(out_degree)):
        if out_degree[i] > 0:
            M.data[M.indptr[i]:M.indptr[i+1]] /= out_degree[i]

    M_T = M.transpose().tocsr()

    pr = np.ones(N) / N

    pool = Pool(processes)

    for _ in range(max_iter):
        pr_old = pr.copy()
        chunk_size = max(1, (N + processes - 1) // processes)
        tasks = [(d, M_T, N, i * chunk_size, min((i + 1) * chunk_size, N), pr_old) for i in range(processes)]

        results = pool.map(pagerank_step, tasks)
        pr = np.concatenate(results)

        if np.linalg.norm(pr - pr_old, 1) < tol:
            break

    pool.close()
    pool.join()

    return {nodes[i]: pr[i] for i in range(N)}

if __name__ == "__main__":
    file_path = r"EX1/web-BerkStan.txt"  

    print("Loading graph...")
    graph = load_graph(file_path)

    print("Calculating PageRank in parallel...")
    pagerank_scores = calculate_pagerank_parallel(graph, processes=4)

    print("Top 10 nodes by PageRank:")
    for node, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Node {node}: {score:.6f}")
