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

	//assume no more than 10 servers/workers
	server_count = static_cast<int>(argv[1][0] - '0');
	worker_count = static_cast<int>(argv[2][0] - '0');
	string file_name(argv[3]);

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
		Server server(world_rank-1, server_count, file_name);
		server.Run();
	}
	else
	{
		//Worker
		Worker worker(server_count);
		worker.LoadFile(file_name, file_name);
		worker.Train();
	}

	MPI_Finalize();
	google::protobuf::ShutdownProtobufLibrary();
	return 0;
}