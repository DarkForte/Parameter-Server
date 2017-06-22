#include<vector>
#include<map>
#include<unordered_map>
#include<iostream>
using std::vector;
using std::unordered_map;
using std::map;
using std::pair;
using std::make_pair;

class Matrix
{
protected:
	vector<map<int, float>> data;


public:
	Matrix() {}

	float Get(int x, int y)
	{
		return data[x][y];
	}

	void AddData(map<int, float> line)
	{
		data.push_back(line);
	}

	bool HasFeature(int line, int feature)
	{
		if (data[line].find(feature) != data[line].end())
			return true;
		else
			return false;
	}

	void print();
	int N() { return data.size(); }

	pair<float, unordered_map<int, float>> CalcLossAndGradient(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int bias_id);
	pair<float, unordered_map<int, float>> CalcLossAndScores(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int bias_id);



};