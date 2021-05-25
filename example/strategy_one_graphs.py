#!/bin/python

def main():
    sorted_edges = []
    max_seq = 0
    with open('graph.txt', 'r') as f:
        for line in f.readlines():
            a, b, c = line.strip().split()
            sorted_edges.append(
                (int(a), int(b), float(c))
            )
            max_seq = max(max_seq, int(a), int(b))

    sorted_edges.sort(reverse=True, key=lambda k:k[2])
    #sorted_edges contains all the edges sorted in descending order based on the edge weight from the original graph.txt file

    #graph will be the networkx graph which will be sparsified with strategy 1 
    for step in range(10, 101, 10):
        edges = []
        for e in sorted_edges[:int(len(sorted_edges)*step/100)]:
            edges.append("{} {} {}\n".format(e[0], e[1], e[2]))
        for ix in range(max_seq+1):
            #adding self edges
            edges.append("{} {} {}\n".format(ix, ix, 1))

        with open("strategy_one_graph_{}_percent.txt".format(step), "w") as f:
            f.writelines(edges)


if __name__ == '__main__':
    main()