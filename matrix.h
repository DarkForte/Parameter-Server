#include<vector>
using std::vector;

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

	Matrix operator * (const Matrix &b) const;

	void print();
	int N() { return data.size(); }
	
};