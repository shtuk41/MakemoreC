#include "pch.h"
#include <iostream>
#include <iomanip>
#include <value.h>

void trace(value& root)
{
	constexpr int  precision = 4;

	std::cout << "{" << root.label() << "," << std::fixed << std::setprecision(precision) << root << "," << root.grad() << "}";

	auto v = root.prev();

	if (v.size() > 0)
	{
		std::cout << "=" << "{" << v[0]->label() << "," << std::fixed << std::setprecision(precision) << *v[0] << "," << v[0]->grad() << "}" << root.op();

		if (root.op().compare("tanh") != 0 &&
			root.op().compare("exp") != 0 &&
			root.op().compare("log") != 0)
		{
			std::cout << "{" << v[1]->label() << "," << *v[1] << "," << v[1]->grad() << "}" << "\n";
			trace(*v[0]);
			trace(*v[1]);
		}
		else
		{
			std::cout << '\n';
			trace(*v[0]);
		}
	}
	else
	{
		std::cout << '\n';
	}
}