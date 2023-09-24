#pragma once

#include <list>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

//bool comp(std::pair<std::pair<char, char>, int> a, std::pair<std::pair<char, char>, int> b) {
//	return a.second < b.second;
//}

class BigramHashFunction
{
public:
	size_t operator()(const std::pair<char, char>& p) const
	{
		return (size_t)(int(p.first) * int(p.second));
	}
};

class LikelihoodHashFunction
{
public:
	size_t operator()(const std::pair<int, int>& p) const
	{
		return (size_t)(p.first * p.second);
	}
};

class Makemore
{
private:
	std::vector<std::string> originalNames;
	std::vector<std::string> names;
	std::list<char> characters;
	std::map<char, int> stoi;
	std::map<int, char> itos;
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction> bigramMap;
	std::vector<std::pair<std::pair<char, char>, int>> bigramVector;
	std::vector<std::vector<int>> bigramDistribution;
	std::unordered_map<std::pair<int,int>, double, LikelihoodHashFunction> logLiklihoodProbabilityMap;
	int** n;
	int ROWS, COLUMNS;

public:

	std::vector<std::string>& Names() { return names; }
	std::vector<std::string>& OriginalNames() { return originalNames; }
	std::list<char>& Characters() { return characters; }
	std::map<char, int>& Stoi() { return stoi; }
	std::map<int, char>& Itos() { return itos; }
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction>& BigramMap() { return bigramMap; };
	std::vector<std::pair<std::pair<char, char>, int>>& BigramVector() { return bigramVector; };

	int** N() { return n; }


	void Init(std::string fileNames, std::optional<int> wordToReads)
	{
		std::ifstream namesFile(fileNames);

		if (namesFile.is_open())
		{
			std::string name;
			std::string single;

			int countWords = 0;

			while (std::getline(namesFile, name))
			{
				originalNames.push_back(name);
				names.push_back(std::string(".") + name + ".");
				single += name;
				
				countWords++;

				if (countWords >= wordToReads.value_or(1000000))
					break;
			}

			std::set<char> singleSet;

			for (auto ii : single)
			{
				singleSet.insert(ii);
			}

			characters = std::list<char>(singleSet.begin(), singleSet.end());
			characters.sort();

			int count = 0;
			itos.insert(std::pair<int, char>(count, '.'));
			stoi.insert(std::pair<char, int>('.', count++));
			

			for (auto c : characters)
			{
				stoi.insert(std::pair<char, int>(c, count));
				itos.insert(std::pair<int, char>(count, c));

				count++;
			}

			ROWS = COLUMNS = count;

			n = new int* [count];
			for (int i = 0; i < count; i++)
			{
				n[i] = new int[count];

				for (int j = 0; j < count; j++)
					n[i][j] = 1;
			}

			for (auto ii : names)
			{
				for (int jj = 0; jj < ii.length() - 1; jj++)
				{
					auto bigram = std::pair<char, char>(ii[jj], ii[jj + 1]);

					int i = stoi[ii[jj]];
					int k = stoi[ii[jj + 1]];

					n[i][k] += 1;

					auto kk = bigramMap.find(bigram);

					if (kk != bigramMap.end())
					{
						kk->second += 1;
					}
					else
					{
						bigramMap[bigram] = 1;
					}
				}
			}

			bigramVector = std::vector<std::pair<std::pair<char, char>, int>>(bigramMap.begin(), bigramMap.end());

			std::sort(bigramVector.rbegin(), bigramVector.rend(), [](std::pair<std::pair<char, char>, int> a, std::pair<std::pair<char, char>, int> b) {return a.second < b.second; });
		
			for (int ii = 0; ii < ROWS; ii++)
			{
				std::vector<int> rowDistributionList(n[ii], n[ii] + COLUMNS);
				bigramDistribution.push_back(rowDistributionList);

				int sum = 0;

				for (auto oo : rowDistributionList)
				{
					sum += oo;
				}

				for (int rr = 0; rr < COLUMNS; rr++)
				{
					logLiklihoodProbabilityMap[std::pair<int, int>(ii, rr)] = (double)rowDistributionList[rr] / double(sum);
				}
				
			}
		}
	}

	~Makemore()
	{
		for (int ii = 0; ii < characters.size() + 1; ii++)
		{
			delete[] n[ii];
		}

		delete[] n;
	}

	char Itos(int num)
	{
		for (auto ii : stoi)
		{
			if (ii.second == num)
				return ii.first;
		}

		throw std::exception(std::string(std::string("itos, num not found: ") + std::to_string(num)).c_str());
	}

	int SampleRow(int rowNumber)
	{
		std::random_device rd;
		auto a = rd();
		std::mt19937 gen(a);

		//std::cout << "row: " << rowNumber << std::endl;

		auto rowVec = bigramDistribution[rowNumber];

		/*for (auto i : rowVec)
		{
			std::cout << i << ',';
		}
		std::cout << '\n';*/

		std::discrete_distribution<int> d(rowVec.begin(), rowVec.end());

		int ret = d(gen);
		return ret;
	}

	std::string GetNameBySampling()
	{
		int ix = 0;
		std::string out;

		while (true)
		{
			ix = SampleRow(ix);
			char o = Itos(ix);
			out += o;

			if (ix == 0)
				break;
		}

		return out;

	}

	double Probability(int row, int col)
	{
		return logLiklihoodProbabilityMap[std::pair<int, int>(row, col)];
	}
};
