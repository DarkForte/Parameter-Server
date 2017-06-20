#include "worker.h"
#include <fstream>
#include <cmath>
using std::swap;

Worker::Worker(int server_count)
{
	this->server_count = server_count;
}

void Worker::LoadFile(string path, string file_name, string label_name)
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

void Worker::Train(int batch_size, int iter_num)
{
	for (int i = 1; i <= iter_num; i++)
	{
		vector<int> minibatch_index = TakeMinibatch(batch_size);
		unordered_map<int, float> param_map = RequestParams(minibatch_index);
		unordered_map<int, float> gradient_map = x.CalcGradient(param_map, minibatch_index, y, feature_num);

		SendGradientMap(gradient_map);
	}

	return;
}

vector<int> Worker::TakeMinibatch(int batch_size)
{
	int n = x.N();
	vector<int> seq(n);
	for (int i = 0; i < n; i++)
		seq[i] = i;

	for (int i = 0; i < batch_size; i++)
	{
		int now_index = rand() % (n - i) + i;
		swap(seq[i], seq[now_index]);
	}

	vector<int> ret(batch_size);
	for (int i = 0; i < batch_size; i++)
		ret[i] = seq[i];

	return ret;

}


unordered_map<int, float> Worker::RequestParams(vector<int> minibatch_index)
{
	vector<vector<int>> feature_requests(server_count);
	int request_sum = 0;

	for (int feature_i = 0; feature_i < feature_num; feature_i++)
	{
		for (int batch_i : minibatch_index)
		{
			if (x.Get(batch_i, feature_i) != 0)
			{
				int server_id = FindWhichServer(feature_i);
				feature_requests[server_id].push_back(feature_i);
				request_sum++;
				break;
			}
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

		MPISendLite(req, server_num + 1, MessageType::PARAM_REQUEST);
	}


	int received_params = 0;
	unordered_map<int, float> param_map;
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

void Worker::SendGradientMap(unordered_map<int, float> gradient_map)
{
	vector<unordered_map<int, float>> gradient_maps(server_count);

	for (auto entry : gradient_map)
	{
		int server_num = FindWhichServer(entry.first);
		gradient_maps[server_num][entry.first] = entry.second;
	}

	for (int server_i = 0; server_i < server_count; server_i++)
	{
		if (gradient_maps[server_i].empty())
			continue;

		unordered_map<int, float> now_map = gradient_maps[server_i];

		ParamServer::GradientRequest gradient_req;
		auto &send_gradient_map = *gradient_req.mutable_gradient_map();
		for (auto entry : now_map)
		{
			send_gradient_map[entry.first] = entry.second;
		}

		MPISendLite(gradient_req, server_i + 1, MessageType::GRADIENT_REQUEST);
	}
	return;
}

int Worker::FindWhichServer(int feature_id)
{
	return feature_id % server_count;
}