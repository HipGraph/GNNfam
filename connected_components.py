import networkx as nx

st = []
end = []


with open('Graph.txt', 'r') as f:
    for line in f.readlines():
        a,b,c = line.strip().split()
        st.append(int(a))
        end.append(int(b))

labels = []
with open('label.txt', 'r') as f:
    for line in f.readlines():
        label = line.strip().split()[1]
        labels.append(int(label))

g = nx.Graph()

for ix, label in enumerate(labels):
    g.add_node(ix, label = label - 1)

for ix in range(len(st)):
    s, e = st[ix], end[ix]
    g.add_edge(s, e)

S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
S.sort(reverse=True, key = lambda k: len(k.nodes))

for k, v in S[0].nodes.items():
    print(k, v)


    
