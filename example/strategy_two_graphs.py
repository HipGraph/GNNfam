#!/home/anuj/virtualenvforest/globalvenv/bin/python


import networkx as nx

def main():
    basegraph = nx.Graph()
    with open('graph.txt', 'r') as f:
        for line in f.readlines():
            a, b, c = line.strip().split()
            basegraph.add_edge(int(a), int(b), weight=float(c))

    #basegraph is the networkx graph generated from the original graph.txt file
    for stepval in range(10, 101, 10):
        step = stepval/100

        all_edges = []
        for node in basegraph.nodes():
            edges = [
                (k,v['weight']) for k,v in sorted(
                    basegraph[node].items(), reverse=True, 
                    key=lambda item:item[1]['weight']
                )
            ]
            keepedges = edges[:int(len(edges)*step)]
            for edge in keepedges:
                all_edges.append(
                    "{} {} {}\n".format(node, edge[0], edge[1])
                )
            all_edges.append("{} {} {}\n".format(node,node, 1))
        with open("strategy_two_graph_{}_percent.txt".format(stepval), "w") as f:
            f.writelines(all_edges)


if __name__ == '__main__':
    main()