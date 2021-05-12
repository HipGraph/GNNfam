### File representations
#### Graph.txt
This file contains the graph structure in the following format, where each line represents an edge between 2 nodes and their edge weight.
These values are space delimited
```
node_a node_b weight_a_b
node_a node_c weight_a_c
.
.
.
```

#### labels.txt
This file contains the labels for each node in the graph. Each line contains space separated values of node name and its class as follows 

```
node_a class__of_node_a
node_b class__of_node_b
.
.
.
```

#### train_test_mask.pkl
This is a python pickle file which contains an array like data structure which is used to create train and test mask for the graph neural network.
The length of the array is equalt to number of nodes in the graph. The array contains integer values 1 and 2. 

Value 1 at index `i` means that node `i` belongs to the training set.

Value 2 at index `i` means that node `i` belongs to the test set.


### Instructions to run
To run use the following command
```
python base_pipeline.py --appropriate_args
```

The default setting can be run by following command

```
python base_pipeline.py --graph Graph.txt --labels label.txt --mask mask_split.pkl --one_indexed_classes
```


### Generating sparsified graphs from protien sequence fasta files.

To start with first step, we will need a fasta file which will look similar to example.fasta file in examples folder.

To create db against which the sequences will be matched with run the following command

```lastdb -p -C 2 lastDB example.fasta```

This will create multiple files which last-align software uses.

After running the first command we run

```
lastal -m 100 -pBLOSUM62 -P 0 -f blasttab lastDB example.fasta
```

This will complete step one and give us sequence alignments which look like the input of step 2 shown in the pipeline image.

Once we have the sequence alignment information, we will need to create a graph input file that looks similar to Graph.txt and similarly a labels.txt file for label information.  
I.e we will need to create a mapping from sequence names to a unique integer for each sequenece. 

Once we have this we can induce sparsity if needed with help of following sample code

```
#---------------------strategy 1 example code-----------------------
#sorted_edges contains all the edges sorted in descending order based on the edge weight from the original graph.txt file

#graph will be the networkx graph which will be sparsified with strategy 1 
for step in range(10, 101, 10):
    name = runname.format(step)
    graph = nx.Graph()
    for e in sorted_edges[:int(len(sorted_edges)*step/100)]:
        graph.add_edge(e[0], e[1])
    print("Doing {} of {} edges".format(len(sorted_edges)*10/100, len(sorted_edges)))
    for ix in range(len(labels)):
        graph.add_edge(ix, ix)

#---------------------strategy 2 example code-----------------------
#basegraph is the networkx graph generated from the original graph.txt file
#gnew will be the networkx graph which will be sparsified with strategy 2
for stepval in range(10, 101, 10):
    name = runname.format(stepval)
    step = stepval/100

    gnew = nx.Graph()
    for node in basegraph.nodes():
        edges = [(k,v['weight']) for k,v in sorted(basegraph[node].items(), reverse=True, key=lambda item:item[1]['weight'])]
        keepedges = edges[:int(len(edges)*step)]
        for edge in keepedges:
            gnew.add_edge(node, edge[0], weight=edge[1])
        gnew.add_edge(node,node,weight=1)
    print("Doing k: {}".format(step))
    print("Edges:", len(gnew.edges()), "Nodes:", len(gnew.nodes()))

#---------------------strategy 3 example code-----------------------
#basegraph is the networkx graph generated from the original graph.txt file
#gnew will be the networkx graph which will be sparsified with strategy 3 
karray = [17, 185, 213, 550, 698, 804]
for keepval in karray:
    gnew = nx.Graph()
    for node in basegraph.nodes():
        edges = [(k,v['weight']) for k,v in sorted(basegraph[node].items(), reverse=True, key=lambda item:item[1]['weight'])]
        keepedges = edges[:keepval]
        for edge in keepedges:
            gnew.add_edge(node, edge[0], weight=1)
        gnew.add_edge(node,node,weight=1)
    print("Doing k: {}".format(keepval))
    print("Edges:", len(gnew.edges()), "Nodes:", len(gnew.nodes()))
```

Once these new sparsified graphs are created you can either edit our pipeline to use these graphs as input or you could save these graphs into new files in similar way Graph.txt file and use our code as is from test_full_pipeline.py  

