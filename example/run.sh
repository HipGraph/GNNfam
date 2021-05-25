#!/bin/bash

echo "Getting graph edges from sequence information...";

lastdb -p -C 2 lastDB example.fasta

echo "Getting graph edges from sequence information..";

lastal -m 100 -pBLOSUM62 -P 0 -f blasttab lastDB example.fasta > sequence_sim_temp.txt

echo "Postprocessing the graph edges...";

awk '{ \
	if ( \
		$1!~/^#/ &&  $3>=30 && $1!=$2 && $14!=0 && $13!=0 && \
		($4-$6)/($13)>=0.7 && ($4-$6)/($14) >=0.7 \
	) print ($1"\t"$2"\t"($3/100))  \
}' sequence_sim_temp.txt > sequence_sim.txt

#update names to indices, stores the mapping in name_to_ix_map.pkl python pickle file
echo "Creating name to integer mapping for each sequence...";
./name_to_index_mapping.py


#create_graphs
echo "Creating graphs with strategy one...";
./strategy_one_graphs.py

echo "Creating graphs with strategy two...";
./strategy_two_graphs.py

echo "Creating graphs with strategy three...";
./strategy_three_graphs.py

echo "Cleaning up temporary files";
#clean up files from last-align
rm lastDB.*
#clean up temporary files
rm sequence_sim_temp.txt



