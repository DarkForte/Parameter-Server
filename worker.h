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
	const std::string path = "/mnt/f/Parameter-Server/";
	Matrix x;
	vector<float> y;
	int feature_num;
	int server_count;

public:
	Worker(int server_count);
	void LoadFile(std::string data_name, std::string label_name = "");
	void Train();
	void Test();
	map<int, float> CalcGradient(map<int, float> param_map);
};