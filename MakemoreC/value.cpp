#include "pch.h"
#include <value.h>

value operator+(float f, value& v)
{
	return v + f;
}

value operator*(float f, value& v)
{
	return v * f;
}

unsigned int value::numberOfOperations = 0;