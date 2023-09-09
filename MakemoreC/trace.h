#pragma once

#include <fstream>
#include "value.h"


void trace(value& root);
void trace_split();
void traceProbability(std::string fileName, const std::vector<std::vector<std::shared_ptr<value>>>& probs)
{
	std::ofstream file(fileName, std::ios::trunc);

	for (auto fd : probs)
	{
		for (auto sd : fd)
		{
			file << std::to_string(*sd) << ",";
		}

		file << "\n";
	}
}

