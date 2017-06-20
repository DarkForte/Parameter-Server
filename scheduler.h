#include<iostream>
#include<fstream>

#include "message.pb.h"
#include "message_type.h"
#include "mpi_helper.h"
#include "constants.h"

using std::cout;
using std::endl;

class Scheduler
{
protected:
	int server_count;
	int worker_count;

public:
	Scheduler(int server_count, int worker_count);
	void Run();
	void TrainingPhase();
	void TestPhase();

protected:
	void CollectWorkerMsg(ParamServer::Command_Type type);
};