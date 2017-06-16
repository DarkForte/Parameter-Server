#include "server.h"
#include "message_type.h"

using std::cout;
using std::vector;
using std::string;

Server::Server(int server_id, int total_servers)
{
	this->server_id = server_id;
	this->total_servers = total_servers;
}

void Server::Run()
{
	using namespace std;
	std::cout << "Server running" << std::endl;

	//Hear from the scheduler the total number of params
	MPI_Recv(&total_params, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	int param_start = total_params / total_servers * (server_id-1);
	int param_num = total_params / total_servers;

	cout << "Total params: " << total_params << endl;

	params.resize(param_num);
	for (int i = 0; i < params.size(); i++)
		params[i] = float(rand()) / RAND_MAX;

	while (true)
	{
		char buf[MAX_LENGTH];

		MPIMsgDescriptor descriptor = MPIRecvLite(buf, MPI_ANY_TAG, MPI_ANY_SOURCE);
		
		MessageType message_type = descriptor.type;
		int source = descriptor.source;
		int length = descriptor.length;

		if (message_type == MessageType::PARAM_REQUEST)
		{
			cout << "Received param request" << endl;
			ParamServer::ParamRequest param_request;
			param_request.ParseFromArray(buf, length);
			HandleParamRequest(param_request, source);
		}
		else if (message_type == MessageType::GRADIENT_REQUEST)
		{
			string in(buf);
			ParamServer::GradientRequest gradient_request;
			gradient_request.ParseFromString(in);
			HandleGradientRequest(gradient_request, source);
		}
	}
}

void Server::HandleParamRequest(ParamServer::ParamRequest req, int source)
{
	int size = req.feature_id_size();
	map<int, float> param_map;

	std::cout << "size: "<< size << std::endl;

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
		cout << entry.first << " " << entry.second << endl;
		response_param_map[entry.first] = entry.second;
	}
	
	MPISendLite(response, source, MessageType::PARAM_RESPONSE);
	return;
}

void Server::HandleGradientRequest(ParamServer::GradientRequest req, int source)
{

}

int Server::ToLocalFeatureID(int global_feature_id)
{
	int num_per_server = total_params / total_servers;
	return global_feature_id % num_per_server;
}