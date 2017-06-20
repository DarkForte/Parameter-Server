#include<mpi.h>
#include<stdio.h>
#include<iostream>

#include "scheduler.h"
#include "server.h"
#include "worker.h"

using namespace std;

int main(int argc, char** argv)
{
	int worker_count = 0;
	int server_count = 0;

	if (argc < 3)
	{
		cout << "Please specify number of servers, workers and the filename." << endl;
		return 0;
	}

	server_count = atoi(argv[1]);
	worker_count = atoi(argv[2]);
	string filepath(argv[3]);
	filepath += "/";
	string filename(argv[4]);
	float learning_rate = atof(argv[5]);
	int iter_nums = atoi(argv[6]);
	int batch_size = atoi(argv[7]);

	GOOGLE_PROTOBUF_VERIFY_VERSION;

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_rank == 0)
	{
		//Scheduler
	}
	else if (world_rank <= server_count)
	{
		//Server
		Server server(world_rank-1, server_count, filepath, filename, learning_rate);
		server.Run();
	}
	else
	{
		//Worker
		Worker worker(server_count);
		worker.LoadFile(filepath, filename, filename);
		worker.Train(batch_size, iter_nums);
	}

	MPI_Finalize();
	google::protobuf::ShutdownProtobufLibrary();
	return 0;
}