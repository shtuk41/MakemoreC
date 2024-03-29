#include "pch.h"
#include <value.h>

value operator+(double f, value& v)
{
	return v + f;
}

value operator*(double f, value& v)
{
	return v * f;
}

unsigned int value::numberOfOperations = 0;
std::vector<std::shared_ptr<value>> value::all_values;
std::vector<std::shared_ptr<value>> value::all_weights;