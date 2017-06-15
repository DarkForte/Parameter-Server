#include "matrix.h"
#include <iostream>

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