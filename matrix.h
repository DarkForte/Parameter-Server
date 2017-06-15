#include<vector>

class Matrix
{
protected:
	std::vector<std::vector<float>> data;

public:
	float Get(int x, int y)
	{
		return data[x][y];
	}

	void AddData(std::vector<float> line)
	{
		data.push_back(line);
	}

	Matrix operator * (const Matrix &b) const;

	void print();
	
};