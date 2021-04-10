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
