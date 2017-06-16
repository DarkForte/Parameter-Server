#include "worker.h"
#include <fstream>

Worker::Worker(int server_count)
{
	this->server_count = server_count;
}

void Worker::LoadFile(std::string file_name, std::string label_name)
{
	using namespace std;
	
	ifstream fin;

	fin.open(path + file_name + ".meta");
	fin >> feature_num;

	cout << "Worker get feature num: " << feature_num << endl;

	fin.close();
	fin.open(path + file_name + ".train");
	vector<float> data(feature_num);

	float tmp;
	int col = 0;
	while (fin >> tmp)
	{
		data[col] = tmp;
		col++;
		if (col == feature_num)
		{
			x.AddData(data);
			col = 0;
		}
	}
	

	if (label_name != "")
	{
		fin.close();
		fin.open(path + label_name);
		
		float tmp;
		while (fin >> tmp)
		{
			y.push_back(tmp);
		}
	}

	//x.print();
	return;
}

void Worker::Train()
{
	vector<vector<int>> feature_requests(server_count+1);
	int request_sum = 0;

	for (int i = 0; i < feature_num; i++)
	{
		if (x.Get(0, i) != 0)
		{
			int server_id = i / 10 +1;
			feature_requests[server_id].push_back(i);
			request_sum++;
		}
	}

	for (int server_num = 1; server_num < feature_requests.size(); server_num++)
	{
		if (feature_requests[server_num].empty())
			continue;

		ParamServer::ParamRequest req;
		for (int feature_id : feature_requests[server_num])
		{
			req.add_feature_id(feature_id);
		}

		/*string out;
		req.SerializeToString(&out);
		MPI_Send(out.data(), out.size(), MPI_CHAR, server_num, static_cast<int>(MessageType::PARAM_REQUEST), MPI_COMM_WORLD);
		*/
		MPISendLite(req, server_num, MessageType::PARAM_REQUEST);
	}

	int received_params = 0;
	map<int, float> param_map;
	while (received_params < request_sum)
	{
		char buf[MAX_LENGTH];
		MPIMsgDescriptor descriptor = MPIRecvLite(buf, MPI_ANY_TAG, MPI_ANY_SOURCE);
		MessageType msg_type = descriptor.type;
		int length = descriptor.length;

		if (msg_type == MessageType::PARAM_RESPONSE)
		{
			ParamServer::ParamResponse param_response;
			param_response.ParseFromArray(buf, length);

			cout << "Worker get response: " << endl;
			for (auto entry : param_response.param_map())
			{
				cout << entry.first << " " << entry.second << endl;
				param_map[entry.first] = entry.second;
			}
			received_params += param_response.param_map_size();
		}
	}

	map<int, float> gradient_map = CalcGradient(param_map);

	ParamServer::GradientRequest gradient_req;
	auto &send_gradient_map = *gradient_req.mutable_gradient_map();
	for (auto entry : gradient_map)
	{
		send_gradient_map[entry.first] = entry.second;
	}

	return;
}

map<int, float> Worker::CalcGradient(map<int, float> param_map)
{
	map<int, float> ret;
	return ret;
}