all:
	g++-11 -fopenmp main.cpp -o main

clean:
	rm -rf ./output/output_*.txt
