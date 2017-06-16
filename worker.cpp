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
	using std::vector;
	using std::string;
	using std::cout;
	using std::endl;

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

	cout << "Feature request for server #1: " << endl;
	for (auto i : feature_requests[1])
		cout << i << " " << endl;

	for (int server_num = 1; server_num < feature_requests.size(); server_num++)
	{
		if (feature_requests[server_num].empty())
			continue;

		ParamServer::ParamRequest req;
		for (int feature_id : feature_requests[server_num])
		{
			req.add_feature_id(feature_id);
		}

		string out;
		req.SerializeToString(&out);
		MPI_Send(out.data(), out.size(), MPI_CHAR, server_num, static_cast<int>(MessageType::PARAM_REQUEST), MPI_COMM_WORLD);
	}

	int received_params = 0;
	while (received_params < request_sum)
	{
		char buf[MAX_LENGTH];
		MPI_Status status;
		MPI_Recv(buf, MAX_LENGTH, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int length;
		MPI_Get_count(&status, MPI_CHAR, &length);
		
		MessageType msg_type = static_cast<MessageType>(status.MPI_TAG);
		if (msg_type == MessageType::PARAM_RESPONSE)
		{
			ParamServer::ParamResponse param_response;
			param_response.ParseFromArray(buf, length);

			cout << "Worker get response: " << endl;
			for (auto entry : param_response.param_map())
			{
				cout << entry.first << " " << entry.second << endl;
			}
			received_params += param_response.param_map_size();
		}
	}
	return;
}