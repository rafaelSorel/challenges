import collections
import os

def traverseGraph(node, graph, visited, stackTraversal):
    """
    No way to use recursive function with 1 million node
    Anyway it is not recommanded at all. may blow out the stack.
    """
    visited[node] = True
    for child in graph[node]:
        if not visited[child]:
            traverseGraph(child, graph, visited, stackTraversal)

    stackTraversal.append(node)

def extract_scc():
    """
    Extract SCC based on Kosaraju algo.
    No rescursive call is used as the there are almost 5_200_000_000 Edges
    For about 1 Million nodes.
    """
    f = open(os.path.join(os.getcwd(), "python/scc.txt"), 'r')
    lines = f.readlines()
    f.close()
    graph = collections.defaultdict(list)
    visited = {}
    for headNode, tailNode in map(lambda x: (int(x[0]), int(x[1])), map(lambda x: x.rstrip().split(' '), lines)):
        graph[headNode].append(tailNode)
        visited[headNode] = False
        if tailNode not in graph:
            graph[tailNode] = list()
            visited[tailNode] = False

    print(f"graph size: {len(graph)}")

    #traverse graph nodes and save the latest visited nodes
    stackTraversal = []
    for nd in filter(lambda x: not visited[x] ,graph):
        pathStack = [nd]
        while len(pathStack) > 0:
            node = pathStack[-1]
            visited[node] = True
            leaf=True
            for child in filter(lambda x: not visited[x], graph[node]):
                leaf=False
                pathStack.append(child)
            if leaf:
                pathStack.pop()
                stackTraversal.append(node)

    print(f"stackTraversal size: {len(stackTraversal)}")
    #transpose graph nodes
    tr_graph = collections.defaultdict(list)
    for node, childs in graph.items():
        for child in childs:
            tr_graph[child].append(node)
            visited[child] = False
        if node not in tr_graph:
            tr_graph[node] = list()
            visited[node] = False

    print(f"tr_graph size: {len(tr_graph)}")

    #DFS on transposed graph based on stackTraversal
    sccs=[]
    while len(stackTraversal) > 0:
        stackNode = stackTraversal.pop()
        if visited[stackNode]:
            continue
        stackQueue = [stackNode]
        scc_length = 0
        while len(stackQueue) > 0:
            node = stackQueue.pop()
            if visited[node]:
                continue
            scc_length+=1
            visited[node] = True
            for child in filter(lambda x: not visited[x], tr_graph[node]):
                stackQueue.append(child)

        if scc_length > 0:
            sccs.append(scc_length)

    sccs = sorted(sccs, reverse=True)
    print ("Top ten biggest SCC:", sccs[:10])

if __name__ == "__main__":
    extract_scc()