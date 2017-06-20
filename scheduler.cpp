#include "scheduler.h"

Scheduler::Scheduler(int server_count, int worker_count)
{
	this->server_count = server_count;
	this->worker_count = worker_count;
}

void Scheduler::Run()
{
	cout << "Scheduler Running" << endl;

	TrainingPhase();
	cout << "Scheduler: Training Phase Finished!" << endl;

	TestPhase();
	cout << "Scheduler: Test Phase Finished!" << endl;
	return;

}

void Scheduler::TrainingPhase()
{
	CollectWorkerMsg(ParamServer::Command_Type::Command_Type_TRAINING_COMPLETE);

	int worker_start = server_count+1;
	for (int i = worker_start; i < worker_start + worker_count; i++)
	{
		ParamServer::Command command;
		command.set_type(ParamServer::Command_Type::Command_Type_START_TEST);
		MPISendLite(command, i, MessageType::COMMAND);
	}
}

void Scheduler::TestPhase()
{
	CollectWorkerMsg(ParamServer::Command_Type::Command_Type_TEST_COMPLETE);

	for (int i = 1; i <= server_count; i++)
	{
		ParamServer::Command command;
		command.set_type(ParamServer::Command_Type::Command_Type_SHUTDOWN);
		MPISendLite(command, i, MessageType::COMMAND);
	}
}

void Scheduler::CollectWorkerMsg(ParamServer::Command_Type type)
{
	int train_finished_workers = 0;
	while (train_finished_workers < worker_count)
	{
		char buffer[MAX_LENGTH];
		MPIMsgDescriptor descriptor = MPIRecvLite(buffer, static_cast<int>(MessageType::COMMAND), MPI_ANY_SOURCE);
		int source = descriptor.source;
		int length = descriptor.length;

		ParamServer::Command command;
		command.ParseFromArray(buffer, length);
		if (command.type() == type)
		{
			train_finished_workers++;
		}
	}

	return;
}