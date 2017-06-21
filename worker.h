#include<iostream>
#include<vector>
#include<map>

#include "matrix.h"
#include "message.pb.h"
#include "message_type.h"
#include "constants.h"
#include "mpi_helper.h"

using std::cout;
using std::string;
using std::vector;
using std::unordered_map;
using std::endl;

class Worker
{
protected:
	Matrix x;
	vector<int> y;
	int feature_num;
	int server_count;

public:
	Worker(int server_count);
	void LoadFile(string path, string data_name);
	void Train(int batch_size, int iter_num);
	void Test();
	void WaitTestCommand();

protected:
	unordered_map<int, float> RequestParams(vector<int> minibatch_index);
	void SendGradientMap(unordered_map<int, float> gradient_map);
	int FindWhichServer(int feature_num);
	vector<int> TakeMinibatch(int batch_size);
	unordered_map<int, float> RequestAllParams();
	unordered_map<int, float> ProposeRequestToServers(vector<vector<int>> feature_requests, int request_sum);
};