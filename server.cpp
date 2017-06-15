#include "server.h"
#include "message_type.h"

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
	int total_params;
	MPI_Recv(&total_params, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	int param_start = total_params / total_servers * (server_id-1);
	int param_num = total_params / total_servers;

	std::cout << "Total params: " << total_params << std::endl;

	params.resize(param_num);
	for (int i = 0; i < params.size(); i++)
		params[i] = float(rand()) / RAND_MAX;

	char buf[MAX_LENGTH] = {};
	while (true)
	{
		MPI_Status message_status;
		MPI_Recv(buf, MAX_LENGTH, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message_status);
		
		MessageType message_type = static_cast<MessageType>(message_status.MPI_TAG);
		int source = message_status.MPI_SOURCE;

		if (message_type == MessageType::PARAM_REQUEST)
		{
			cout << "Received param request" << endl;
			string in(buf);
			ParamServer::ParamRequest param_request;
			cout << in.length() << endl;
			param_request.ParseFromString(in);
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
	using std::map;
	using std::string;

	int size = req.feature_id_size();
	map<int, float> param_map;

	std::cout << "size: "<< size << std::endl;

	for (int i = 0; i < size; i++)
	{
		int now_feature_id = req.feature_id(i);
		int local_feature_id = ToLocalFeatureID(now_feature_id);

		std::cout << "now feature id: " << now_feature_id << std::endl;
		param_map[now_feature_id] = params[local_feature_id];
	}

	ParamServer::ParamResponse response;
	auto& response_param_map = *response.mutable_param_map();
	for (auto entry : param_map)
	{
		std::cout << entry.first << " " << entry.second << std::endl;
		response_param_map[entry.first] = entry.second;
	}
	string out;
	response.SerializeToString(&out);
	MPI_Send(out.c_str(), out.length(), MPI_CHAR, source, static_cast<int>(MessageType::PARAM_RESPONSE), MPI_COMM_WORLD);
	return;
}

void Server::HandleGradientRequest(ParamServer::GradientRequest req, int source)
{

}

int Server::ToLocalFeatureID(int global_feature_id)
{
	int num_per_server = global_feature_id / total_servers;
	return global_feature_id % num_per_server;
}