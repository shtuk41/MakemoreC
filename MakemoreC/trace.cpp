#include "pch.h"
#include <iostream>
#include <iomanip>
#include <value.h>
#include <fstream>

void trace(value& root)
{
	std::ofstream traceFile("trace.txt", std::ios::app);

	constexpr int  precision = 4;

	traceFile << "{" << root.label() << "," << std::fixed << std::setprecision(precision) << std::to_string(root) << "," << std::to_string(root.grad()) << "}";

	auto v = root.prev();

	if (v.size() > 0)
	{
		traceFile << "=" << "{" << v[0]->label() << "," << std::fixed << std::setprecision(precision) << std::to_string(*v[0]) << "," << std::to_string(v[0]->grad()) << "}" << root.op();

		if (root.op().compare("tanh") != 0 &&
			root.op().compare("exp") != 0 &&
			root.op().compare("log") != 0)
		{
			traceFile << "{" << v[1]->label() << "," << std::to_string(*v[1]) << "," << std::to_string(v[1]->grad()) << "}" << "\n";
			trace(*v[0]);
			trace(*v[1]);
		}
		else
		{
			traceFile << '\n';
			trace(*v[0]);
		}
	}
	else
	{
		traceFile << '\n';
	}
}