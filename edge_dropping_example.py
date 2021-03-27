import networkx as nx 
import numpy as np 

seedval = 42
np.random.seed(seedval)

basegraph = nx.karate_club_graph() 
nx.draw_circular(basegraph, with_labels=True, seed=seedval)  
plt.savefig('karate_club_graph.png') 
plt.close('all') 

for node in basegraph.nodes():  
    for e in basegraph[node]:  
        wt = random.randint(50, 100)/100  
        basegraph[node][e]['weight'] = wt  
        basegraph[e][node]['weight'] = wt  
 
per_node_edge_cut_off = int(
    sum([s[1] for s in basegraph.degree()])/len(basegraph.degree())
) - 1  
 
gnew = nx.Graph() 
 
for node in basegraph.nodes(): 
    edges = [(k,v['weight']) for k,v in sorted(basegraph[node].items(), reverse=True, key=lambda item:item[1]['weight'])] 
    keepedges = edges[:per_node_edge_cut_off] 
    for edge in keepedges: 
        gnew.add_edge(node, edge[0], weight=1) 
    gnew.add_edge(node,node,weight=1) 
 
nx.draw_circular(gnew, with_labels=True, seed=seedval)  
plt.savefig(
    'karate_club_graph_per_node_drop_k_{}.png'.format(per_node_edge_cut_off)
) 
plt.close('all')

edges = []

for st, end in basegraph.edges():
    edges.append(((st, end), basegraph[st][end]))

gnew = nx.Graph() 
edges.sort(reverse=True, key=lambda k:k[1]['weight'])
edges.sort(reverse=True, key=lambda k:k[1]['weight']) 
for edge in edges[:int(len(edges)*0.5)]: 
    st, end = edge[0] 
    wt = edge[1]['weight'] 
    gnew.add_edge(st, end,weight=wt)
nx.draw_circular(gnew, with_labels=True, seed=seedval)  
plt.savefig('karate_club_graph_global_half_drop.png')
plt.close('all')

gnew = nx.Graph() 
for node in basegraph.nodes():
    edges = [(
        k,v['weight']) for k,v in sorted(basegraph[node].items(), 
        reverse=True, key=lambda item:item[1]['weight']
    )]
    keepedges = edges[:int(len(edges)*0.5)]
    for edge in keepedges:
        gnew.add_edge(node, edge[0], weight=edge[1])
nx.draw_circular(gnew, with_labels=True, seed=seedval)  
plt.savefig('karate_club_graph_local_half_drop.png')
plt.close('all')
