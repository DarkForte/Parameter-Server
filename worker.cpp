#include "worker.h"
#include <fstream>
#include <sstream>
#include <cmath>
using std::swap;

Worker::Worker(int server_count)
{
	this->server_count = server_count;
}

void Worker::LoadFile(string path, string file_name)
{
	cout << "Reading file: " << file_name << endl;
	using namespace std;

	ifstream fin;

	fin.open(path + file_name + ".meta");
	fin >> feature_num;

	fin.close();
	fin.open(path + file_name);

	int line = 0;
	string now_line;
	while (getline(fin, now_line))
	{

		istringstream ss(now_line);
		int now_label;
		ss >> now_label;
		y.push_back(now_label);

		map<int, float> now_data;
		string feature_string;
		while (ss >> feature_string)
		{
			istringstream feature_ss(feature_string);
			string buf;
			getline(feature_ss, buf, ':');
			int pos = stoi(buf);
			getline(feature_ss, buf, ':');
			float value = stof(buf);

			now_data[pos] = value;

		}

		x.AddData(now_data);
		line++;
	}

	cout << "Read complete" << endl;

	return;
}

void Worker::Train(int batch_size, int iter_num)
{
	for (int i = 1; i <= iter_num; i++)
	{
		vector<int> minibatch_index = TakeMinibatch(batch_size);
		unordered_map<int, float> param_map = RequestParams(minibatch_index);
		unordered_map<int, float> gradient_map = x.CalcGradient(param_map, minibatch_index, y, feature_num);

		for (auto entry : gradient_map)
			cout << entry.first << " " << entry.second << endl;

		SendGradientMap(gradient_map);
	}

	ParamServer::Command command;
	command.set_type(ParamServer::Command_Type::Command_Type_TRAINING_COMPLETE);
	MPISendLite(command, SCHEDULER_ID, MessageType::COMMAND);

	return;
}

void Worker::WaitTestCommand()
{
	char buf[MAX_LENGTH];
	while (true)
	{
		MPIMsgDescriptor descriptor = MPIRecvLite(buf, static_cast<int>(MessageType::COMMAND), SCHEDULER_ID);
		int length = descriptor.length;

		ParamServer::Command command;
		command.ParseFromArray(buf, length);
		if (command.type() == ParamServer::Command_Type::Command_Type_START_TEST)
			break;
	}

	return;
}

void Worker::Test()
{
	cout << "Worker testing" << endl;
	int n = x.N();

	vector<int> index(n);
	for (int i = 0; i < n; i++)
		index[i] = i;

	unordered_map<int, float> param_map = RequestAllParams();
	pair<float, unordered_map<int, float>> loss_and_scores = x.CalcLossAndScores(param_map, index, y, feature_num);
	float loss = loss_and_scores.first;
	auto scores = loss_and_scores.second;

	vector<int> predict(n);
	for (int i = 0; i < n; i++)
	{
		if (scores[i] < 0.5)
			predict[i] = 0;
		else
			predict[i] = 1;
	}

	int correct_count = 0;
	for (int i = 0; i < n; i++)
	{
		if (predict[i] == y[i])
			correct_count++;
	}
	cout << "Correct ratio:" << float(correct_count) / float(x.N()) << endl;

	ParamServer::Command command;
	command.set_type(ParamServer::Command_Type::Command_Type_TEST_COMPLETE);
	MPISendLite(command, SCHEDULER_ID, MessageType::COMMAND);

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

unordered_map<int, float> Worker::RequestAllParams()
{
	vector<vector<int>> feature_requests(server_count);
	int request_sum = 0;
	for (int i = 0; i < feature_num + 1; i++)
	{
		int server_id = FindWhichServer(i);
		feature_requests[server_id].push_back(i);
	}

	request_sum = feature_num + 1;
	cout << "AllParams request complete, sum= " << request_sum << endl;
	return ProposeRequestToServers(feature_requests, request_sum);

}

unordered_map<int, float> Worker::RequestParams(vector<int> minibatch_index)
{
	vector<vector<int>> feature_requests(server_count);
	int request_sum = 0;

	for (int feature_i = 0; feature_i < feature_num; feature_i++)
	{
		for (int batch_i : minibatch_index)
		{
			if (x.HasFeature(batch_i, feature_i))
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

	cout << "Request sum:" << request_sum << endl;

	auto param_map = ProposeRequestToServers(feature_requests, request_sum);

	return param_map;
}

unordered_map<int, float> Worker::ProposeRequestToServers(vector<vector<int>> feature_requests, int request_sum)
{
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

		cout << "Receive Params: " << received_params << endl;
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