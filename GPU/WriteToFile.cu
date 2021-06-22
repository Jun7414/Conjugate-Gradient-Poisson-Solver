void WriteToFile(double *u, int N)
{
	std::ofstream ofs;
        ofs.open("output.txt");
        if (!ofs.is_open()) {
                printf("Failed to open file.\n");
        } else {
                for(int i=0; i<N; i++)
                for(int j=0; j<N; j++){
                        ofs << u[N*i+j] << " ";

                }
                ofs.close();
        }
}

