#include "server.h"
#include "message_type.h"
#include <fstream>

Server::Server(int server_id, int total_servers, string filepath, string filename, double learning_rate, string processor_name)
{
	this->server_id = server_id;
	this->total_servers = total_servers;
	this->filepath = filepath;
	this->filename = filename;
	this->learning_rate = learning_rate;
	this->processor_name = processor_name;
}

void Server::Run()
{
	cout << "Server " << server_id <<" running on "<< processor_name << endl;

	std::ifstream fin(filepath + filename + ".meta");
	fin >> total_params;

	int param_with_bias = total_params + 1;

	int param_num = (param_with_bias-1) / total_servers + 1;

	cout << "Total params: " << total_params << endl;
	cout << "Param num on " << server_id << " = " << param_num << endl;

	params.resize(param_num);
	for (int i = 0; i < params.size(); i++)
	{
		params[i] = double(rand()) / RAND_MAX * 2 - 1;
		//cout << params[i] << endl;
	}	//params[i] = 0;
	
	/*params[0] = 0.2;
	params[1] = 0.2;
	params[2] = -24;
	*/

	/*if (server_id == 0)
	{
		params[0] = 0.2;
		params[1] = -24;
	}
	else
		params[0] = 0.2;*/

	while (true)
	{
		char buf[MAX_LENGTH];

		MPIMsgDescriptor descriptor = MPIRecvLite(buf, MPI_ANY_TAG, MPI_ANY_SOURCE);
		
		MessageType message_type = descriptor.type;
		int source = descriptor.source;
		int length = descriptor.length;

		if (message_type == MessageType::PARAM_REQUEST)
		{
			//cout << "Received param request" << endl;
			ParamServer::ParamRequest param_request;
			param_request.ParseFromArray(buf, length);
			HandleParamRequest(param_request, source);
		}
		else if (message_type == MessageType::GRADIENT_REQUEST)
		{
			//cout << "Received gradient request" << endl;
			ParamServer::GradientRequest gradient_request;
			gradient_request.ParseFromArray(buf, length);
			HandleGradientRequest(gradient_request, source);
		}
		else if (message_type == MessageType::COMMAND)
		{
			ParamServer::Command command;
			command.ParseFromArray(buf, length);

			if (command.type() == ParamServer::Command_Type::Command_Type_SHUTDOWN)
				break;
		}
	}
	return;
}

void Server::HandleParamRequest(ParamServer::ParamRequest req, int source)
{
	int size = req.feature_id_size();
	unordered_map<int, double> param_map;

	for (int i = 0; i < size; i++)
	{
		int now_feature_id = req.feature_id(i);
		int local_feature_id = ToLocalFeatureID(now_feature_id);

		param_map[now_feature_id] = params[local_feature_id];
	}

	ParamServer::ParamResponse response;
	auto& response_param_map = *response.mutable_param_map();
	for (auto entry : param_map)
	{
		//cout << entry.first << " " << entry.second << endl;
		response_param_map[entry.first] = entry.second;
	}
	
	MPISendLite(response, source, MessageType::PARAM_RESPONSE);
	return;
}

void Server::HandleGradientRequest(ParamServer::GradientRequest req, int source)
{
	for (auto entry : req.gradient_map())
	{
		//cout << entry.first << " " << entry.second << endl;

		int local_id = ToLocalFeatureID(entry.first);
		params[local_id] -= learning_rate * entry.second;
	}
}

int Server::ToLocalFeatureID(int global_feature_id)
{
	return global_feature_id / total_servers;
}