#include "matrix.h"
#include <iostream>
#include<cmath>
using std::swap;
using std::cout;
using std::endl;

void Matrix::print()
{
	using namespace std;
	for (vector<float> line : data)
	{
		for (float i : line)
		{
			cout << i << " ";
		}
		cout << endl;
	}
}

unordered_map<int, float> Matrix::CalcGradient(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int feature_num)
{
	unordered_map<int, float> ret;

	int n = batch_index.size();
	unordered_map<int, float> scores;

	float loss = 0;

	for (int i : batch_index)
	{
		float sum = 0;
		for (int j = 0; j < feature_num; j++)
		{
			sum += Get(i, j) * param_map[j];
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

	for (int i : batch_index)
	{
		float d_sum = scores[i] - y[i];

		for (int j = 0; j < feature_num; j++)
		{
			ret[j] += d_sum * Get(i, j) / n;
		}

		//bias
		ret[feature_num] += d_sum / n;
	}

	return ret;
}