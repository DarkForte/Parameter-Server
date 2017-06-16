#include "worker.h"
#include <fstream>
#include <cmath>

Worker::Worker(int server_count)
{
	this->server_count = server_count;
}

void Worker::LoadFile(std::string file_name, string label_name)
{
	using namespace std;
	
	ifstream fin;

	fin.open(path + file_name + ".meta");
	fin >> feature_num;

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
		fin.open(path + label_name + ".label");
		
		int tmp;
		while (fin >> tmp)
		{
			y.push_back(tmp);
		}
		fin.close();
	}

	cout << "y size: " << y.size() << endl;

	return;
}

void Worker::Train()
{
	for (int i = 1; i <= 3000; i++)
	{
		map<int, float> param_map = RequestParams();
		map<int, float> gradient_map = CalcGradient(param_map);

		SendGradientMap(gradient_map);
	}

	return;
}

map<int, float> Worker::CalcGradient(map<int, float> param_map)
{
	map<int, float> ret;

	int n = x.N();
	vector<float> scores(n);

	float loss = 0;

	for (int i = 0; i < n; i++)
	{
		float sum = 0;
		for (int j = 0; j < feature_num; j++)
		{
			sum += x.Get(i, j) * param_map[j];
		}
		//bias
		sum += param_map[feature_num];

		float score = 1 / (1 + exp(-sum));
		scores[i] = score;
		if (y[i] == 1)
		{
			loss += -log(score);
		}
		else
		{
			loss += -log(1 - score);
		}
	}
	loss /= n;

	cout << "loss: " << loss << endl;

	for (int i = 0; i < n; i++)
	{
		float d_sum = scores[i] - y[i];

		for (int j = 0; j < feature_num; j++)
		{
			ret[j] += d_sum * x.Get(i, j) / n;
		}

		//bias
		ret[feature_num] += d_sum / n;
	}

	return ret;
}

map<int, float> Worker::RequestParams()
{
	vector<vector<int>> feature_requests(server_count);
	int request_sum = 0;

	for (int i = 0; i < feature_num; i++)
	{
		if (x.Get(0, i) != 0)
		{
			int server_id = FindWhichServer(i);
			feature_requests[server_id].push_back(i);
			request_sum++;
		}
	}

	//bias
	feature_requests[FindWhichServer(feature_num)].push_back(feature_num);
	request_sum++;

	for (int server_num = 0; server_num < server_count; server_num++)
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
		MPISendLite(req, server_num+1, MessageType::PARAM_REQUEST);
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

			for (auto entry : param_response.param_map())
			{
				param_map[entry.first] = entry.second;
			}
			received_params += param_response.param_map_size();
		}
	}

	return param_map;
}

void Worker::SendGradientMap(map<int, float> gradient_map)
{
	vector<map<int, float>> gradient_maps(server_count);

	for (auto entry : gradient_map)
	{
		int server_num = FindWhichServer(entry.first);
		gradient_maps[server_num][entry.first] = entry.second;
	}

	for (int server_i=0; server_i < server_count; server_i++)
	{
		if (gradient_maps[server_i].empty())
			continue;

		map<int, float> now_map = gradient_maps[server_i];

		ParamServer::GradientRequest gradient_req;
		auto &send_gradient_map = *gradient_req.mutable_gradient_map();
		for (auto entry : now_map)
		{
			send_gradient_map[entry.first] = entry.second;
		}

		MPISendLite(gradient_req, server_i+1, MessageType::GRADIENT_REQUEST);
	}
	return;
}

int Worker::FindWhichServer(int feature_id)
{
	return feature_id % server_count;
}