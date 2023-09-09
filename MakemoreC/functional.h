#pragma once

#include <array>

template <std::size_t numberOfClasses>
std::array<double, numberOfClasses> one_hot(int num)
{
	std::array<double, numberOfClasses> hot;

	for (auto it = hot.begin(); it != hot.end(); ++it)
	{
		*it = 0;
	}

	hot[num] = 1.0f;

	return hot;
}

template <std::size_t numberOfInputs, std::size_t numberOfClasses>
std::array<std::array<double, numberOfClasses>, numberOfInputs> one_hot(const std::array<int, numberOfInputs> & input)
{
	std::array<std::array<double, numberOfClasses>, numberOfInputs> hotOuput;

	int count = 0;

	for (auto ii : input)
	{
		std::array<double, numberOfClasses> hot;

		for (auto it = hot.begin(); it != hot.end(); ++it)
		{
			*it = 0.0f;
		}

		if (ii >= numberOfClasses)
		{
			throw std::exception("out of bound in one_hot");
		}

		hot[ii] = 1.0f;

		hotOuput[count] =  hot;

		count++;
	}

	return hotOuput;
}
