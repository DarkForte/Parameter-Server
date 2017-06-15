#include<mpi.h>
#include<iostream>
#include<vector>

#include "matrix.h"
#include "message.pb.h"
#include "message_type.h"
#include "constants.h"

class Worker
{
protected:
	const std::string path = "/mnt/f/Parameter-Server/";
	Matrix x;
	std::vector<float> y;
	int feature_num;
	int server_count;

public:
	Worker(int server_count);
	void LoadFile(std::string data_name, std::string label_name = "");
	void Train();
	void Test();
};