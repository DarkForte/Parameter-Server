#include<mpi.h>
#include<stdio.h>
#include<iostream>

#include "scheduler.h"
#include "server.h"
#include "worker.h"
#include "mpi_helper.h"

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
	string filename_train(argv[4]);
	string filename_test(argv[5]);
	float learning_rate = atof(argv[6]);
	int iter_num = atoi(argv[7]);
	int batch_size = atoi(argv[8]);
	int seed = atoi(argv[9]);

	srand(seed);

	GOOGLE_PROTOBUF_VERIFY_VERSION;

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	string processor_name = GetProcessorName();

	if (world_rank == 0)
	{
		//Scheduler
		Scheduler scheduler(server_count, worker_count);
		scheduler.Run();
	}
	else if (world_rank <= server_count)
	{
		//Server
		Server server(world_rank-1, server_count, filepath, filename_train, learning_rate, processor_name);
		server.Run();
	}
	else
	{
		//Worker
		Worker train_worker(server_count, processor_name);
		train_worker.LoadFile(filepath, filename_train);
		train_worker.Train(batch_size, iter_num);

		Worker test_worker(server_count, processor_name);
		test_worker.LoadFile(filepath, filename_test);
		test_worker.WaitTestCommand();
		test_worker.Test();
	}

	MPI_Finalize();
	google::protobuf::ShutdownProtobufLibrary();
	return 0;
}