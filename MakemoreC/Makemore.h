#pragma once

#include <list>
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

class Makemore
{
private:
	std::vector<std::string> originalNames;
	std::vector<std::string> names;
	std::list<char> characters;
	std::map<char, int> stoi;
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction> bigramMap;
	std::vector<std::pair<std::pair<char, char>, int>> bigramVector;
	int** n;

public:
	void Init(std::string fileNames)
	{
		std::ifstream namesFile(fileNames);

		if (namesFile.is_open())
		{
			std::string name;
			std::string single;
			while (std::getline(namesFile, name))
			{
				originalNames.push_back(name);
				names.push_back(std::string("@") + name + "$");
				single += name;
			}

			std::set<char> singleSet;

			for (auto ii : single)
			{
				singleSet.insert(ii);
			}

			characters = std::list<char>(singleSet.begin(), singleSet.end());
			characters.sort();

			int count = 0;
			for (auto c : characters)
			{
				stoi.insert(std::pair<char, int>(c, count));
				count++;
			}

			stoi.insert(std::pair<char, int>('@', count++));
			stoi.insert(std::pair<char, int>('$', count++));

			n = new int* [count];
			for (int i = 0; i < count; i++)
			{
				n[i] = new int[count] {0};
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

	std::vector<std::string>& Names() { return names; }
	std::vector<std::string>& OriginalNames() { return originalNames; }
	std::list<char>& Characters() { return characters; }
	std::map<char, int>& Stoi() { return stoi; }
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction>& BigramMap() { return bigramMap; };
	std::vector<std::pair<std::pair<char, char>, int>>& BigramVector() { return bigramVector; };

	int** N() { return n; }

};
