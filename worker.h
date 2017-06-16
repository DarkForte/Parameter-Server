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
using std::map;
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
	void LoadFile(std::string data_name, std::string label_name = "");
	void Train();
	void Test();

protected:
	map<int, float> CalcGradient(map<int, float> param_map);
	map<int, float> RequestParams();
	void SendGradientMap(map<int, float> gradient_map);
	int FindWhichServer(int feature_num);
};