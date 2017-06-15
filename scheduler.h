#include<mpi.h>
#include<iostream>
#include<fstream>

class Scheduler
{
protected:
	//const std::string path = "/home/keweichen/Parameter-Server/";
	const std::string path = "/mnt/f/Parameter-Server/";
	std::string filename;
	int feature_count;
	int server_count;
	int worker_count;

public:
	Scheduler(std::string filename, int server_count, int worker_count);
	void Run();

};