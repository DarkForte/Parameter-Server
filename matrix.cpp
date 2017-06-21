#include "matrix.h"
#include <iostream>
#include<cmath>
using std::swap;
using std::cout;
using std::endl;

void Matrix::print()
{
	using namespace std;
	for (auto line : data)
	{
		for (auto entry : line)
		{
			cout << entry.first << ":" << entry.second;
		}
		cout << endl;
	}
}

pair<float, unordered_map<int, float>> Matrix::CalcLossAndScores(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int bias_id)
{
	int n = batch_index.size();
	unordered_map<int, float> scores;

	float loss = 0;

	for (int i : batch_index)
	{
		float sum = 0;
		for (auto entry : data[i])
		{
			int pos = entry.first;
			float value = entry.second;

			sum += param_map[pos] * value;
		}
		//bias
		sum += param_map[bias_id];

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

	return make_pair(loss, scores);
}


unordered_map<int, float> Matrix::CalcGradient(unordered_map<int, float> param_map, vector<int> batch_index, vector<int> y, int bias_id)
{
	unordered_map<int, float> ret;

	int n = batch_index.size();

	auto loss_and_scores = CalcLossAndScores(param_map, batch_index, y, bias_id);
	float loss = loss_and_scores.first;
	auto scores = loss_and_scores.second;

	cout << "loss: " << loss << endl;

	for (int i : batch_index)
	{
		float d_sum = scores[i] - y[i];

		/*for (int j = 0; j < feature_num; j++)
		{
			ret[j] += d_sum * Get(i, j) / n;
		}*/
		for (auto entry : data[i])
		{
			int pos = entry.first;
			int value = entry.second;
			ret[pos] += d_sum * value / n;
		}

		//bias
		ret[bias_id] += d_sum / n;
	}

	return ret;
}