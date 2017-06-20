#include<vector>
#include<unordered_map>
#include<iostream>
using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;

class Matrix
{
protected:
	vector<vector<float>> data;


public:
	Matrix() {}
	Matrix(int n, int m)
	{
		data.resize(n, vector<float>(m));
	}

	float Get(int x, int y)
	{
		return data[x][y];
	}

	void Set(int x, int y, int value)
	{
		data[x][y] = value;
	}

	void AddData(std::vector<float> line)
	{
		data.push_back(line);
	}

	void print();
	int N() { return data.size(); }

	unordered_map<int, float> CalcGradient(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int feature_num);
	pair<float, unordered_map<int, float>> CalcLossAndScores(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int feature_num);



};