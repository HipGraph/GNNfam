#!/bin/python
import pickle

def main():
	name_to_ix_map = {}
	ix = 0
	lines = []

	with open("sequence_sim.txt", "r") as f:
		for line in f.readlines():
			a, b, c = line.strip().split()
			if a not in name_to_ix_map:
				name_to_ix_map[a] = ix
				ix+=1
			if b not in name_to_ix_map:
				name_to_ix_map[b] = ix
				ix+=1

			lines.append(
				"{seq_a} {seq_b} {edge_wt}\n".format(
					seq_a=name_to_ix_map[a],
					seq_b=name_to_ix_map[b],
					edge_wt=c
				)
			)

	#save the map
	with open("name_to_ix_map.pkl", "wb") as f:
		pickle.dump(name_to_ix_map, f)

	#create updated graph.txt		
	with open("graph.txt", "w") as f:
		f.writelines(lines)


if __name__ == '__main__':
	main()