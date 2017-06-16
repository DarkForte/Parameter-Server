#include "scheduler.h"

Scheduler::Scheduler(std::string filename, int server_count, int worker_count)
{
	this->filename = filename;
	this->server_count = server_count;
	this->worker_count = worker_count;

}

void Scheduler::Run()
{
	std::cout << "Scheduler Running" << std::endl;

	std::ifstream fin(path + filename + ".meta");
	fin >> feature_count;


	int i;
	for (i = 1; i <= server_count; i++)
	{
		//Tell the servers the number of features
		MPI_Send(&feature_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	}

	for (i = 1; i <= worker_count; i++)
	{
		//Tell the workers to load data and start
		MPI_Send(filename.c_str(), filename.length(), MPI_CHAR, server_count + i, 0, MPI_COMM_WORLD);
	}
	return;

}