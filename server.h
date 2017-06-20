#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>

#include "message.pb.h"
#include "constants.h"
#include "mpi_helper.h"

using std::cout;
using std::vector;
using std::unordered_map;
using std::endl;
using std::string;

class Server
{
protected:
	int server_id;
	int total_servers;
	int total_params;
	string filename;
	vector<float> params;
	float learning_rate = 0.001;

public:
	Server(int server_id, int total_servers, string filename);
	void Run();

protected:
	void HandleParamRequest(ParamServer::ParamRequest req, int source);
	void HandleGradientRequest(ParamServer::GradientRequest req, int source);
	int ToLocalFeatureID(int global_id);
};