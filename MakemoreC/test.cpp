#include "pch.h"
#include <fstream>
#include <iostream>
#include <list>
#include <ranges>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

class Makemore
{
private:
	std::vector<std::string> original;
	std::vector<std::string> names;
	std::list<char> characters;
	std::map<char, int> stoi;
	
public:
	std::vector<std::string> & Names() { return names; }
	std::vector<std::string>& OriginalNames() { return original; }
	std::list<char>& Characters() { return characters; }
	std::map<char, int>& Stoi() { return stoi; }
	int** N() { return n; }
	int** n;
 };

class MakemoreTest : public ::testing::Test 
{
protected:
	Makemore mm;

	void SetUp() override
	{
		std::ifstream namesFile("names.txt");

		if (namesFile.is_open())
		{
			std::string name;
			std::string single;
			while (std::getline(namesFile, name))
			{
				mm.OriginalNames().push_back(name);
				mm.Names().push_back(std::string("@") + name + "$");
				single += name;
			}

			std::set<char> singleSet;

			for (auto ii : single)
			{
				singleSet.insert(ii);
			}

			mm.Characters() = std::list<char>(singleSet.begin(), singleSet.end());
			mm.Characters().sort();

			int count = 0;
			for (auto c : mm.Characters())
			{
				mm.Stoi().insert(std::pair<char, int>(c, count));
				count++;
			}

			mm.Stoi().insert(std::pair<char, int>('@', count++));
			mm.Stoi().insert(std::pair<char, int>('$', count));

			mm.n = new int* [count];
			for (int i = 0; i < count; i++)
			{
				mm.n[i] = new int[count] {0};
			}
		}
	}

	void TearDown() override
	{
		for (int ii = 0; ii < mm.Characters().size() + 1; ii++)
		{
			delete [] mm.n[ii];
		}

		delete [] mm.n;
	}
};

TEST_F(MakemoreTest, ReadNames)
{
	size_t length = mm.Names().size();

	EXPECT_EQ(length, 32033);
}

TEST_F(MakemoreTest, Print10)
{
	auto ten = mm.Names() | std::ranges::views::take(10);
	

	for (auto i : ten)
	{
		std::cout << i << '\n';
	}
}

TEST_F(MakemoreTest, FindMinLength)
{
	auto shortest = std::min_element(mm.OriginalNames().begin(), mm.OriginalNames().end(), [](const std::string& s1, const std::string& s2) {return s1.size() < s2.size(); });

	std::cout << *shortest << std::endl;

	EXPECT_EQ(shortest->size(), 2);
}

TEST_F(MakemoreTest, FindMaxLength)
{
	auto longest = std::min_element(mm.OriginalNames().begin(), mm.OriginalNames().end(), [](const std::string& s1, const std::string& s2) {return s1.size() > s2.size(); });

	std::cout << *longest << std::endl;

	EXPECT_EQ(longest->size(), 15);
}

TEST_F(MakemoreTest, FindTwoLetterPairs)
{
	auto three = mm.Names() | std::ranges::views::take(3);
	std::vector<std::pair<char, char>> two_letter_pairs;

	for (auto ii : three)
	{
		for (int jj = 0; jj < ii.length() - 1; jj++)
		{
			two_letter_pairs.push_back(std::pair<char, char>(ii[jj], ii[jj + 1]));
		}
	}

	for (auto ii : two_letter_pairs)
	{
		std::cout << ii.first << ", " << ii.second << '\n';
	}
}

class BigramHashFunction
{
public:
	size_t operator()(const std::pair<char, char>& p) const
	{
		return (size_t)(int(p.first) * int(p.second));
	}
};

TEST_F(MakemoreTest, BigramStatistics3Words_map)
{
	auto three = mm.Names() | std::ranges::views::take(3);

	std::unordered_map<std::pair<char, char>, int, BigramHashFunction> b;

	for (auto ii : three)
	{
		for (int jj = 0; jj < ii.length() - 1; jj++)
		{
			auto bigram = std::pair<char, char>(ii[jj], ii[jj + 1]);

			auto kk = b.find(bigram);

			if (kk != b.end())
			{
				kk->second += 1;
			}
			else
			{
				b[bigram] = 1;
			}
		}
	}

	for (auto bigram : b)
	{
		std::cout << bigram.first.first << "," << bigram.first.second << '=' << bigram.second << '\n';
	}
}

TEST_F(MakemoreTest, BigramStatisticsAllWords_map)
{
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction> b;

	for (auto ii : mm.Names())
	{
		for (int jj = 0; jj < ii.length() - 1; jj++)
		{
			auto bigram = std::pair<char, char>(ii[jj], ii[jj + 1]);

			auto kk = b.find(bigram);

			if (kk != b.end())
			{
				kk->second += 1;
			}
			else
			{
				b[bigram] = 1;
			}
		}
	}

	for (auto bigram : b)
	{
		std::cout << "new: " << bigram.first.first << "," << bigram.first.second << '=' << bigram.second << '\n';
	}
}

bool comp(std::pair<std::pair<char, char>, int> a, std::pair<std::pair<char, char>, int> b) {
	return a.second < b.second;
}


TEST_F(MakemoreTest, BigramStatisticsAllWords_Sorted)
{
	std::unordered_map<std::pair<char, char>, int, BigramHashFunction> b;

	for (auto ii : mm.Names())
	{
		for (int jj = 0; jj < ii.length() - 1; jj++)
		{
			auto bigram = std::pair<char, char>(ii[jj], ii[jj + 1]);

			auto kk = b.find(bigram);

			if (kk != b.end())
			{
				kk->second += 1;
			}
			else
			{
				b[bigram] = 1;
			}
		}
	}

	std::vector<std::pair<std::pair<char, char>, int>> view = std::vector<std::pair<std::pair<char, char>, int>>(b.begin(), b.end());

	std::sort(view.rbegin(), view.rend(), comp);

	for (auto bigram : view)
	{
		std::cout << bigram.first.first << "," << bigram.first.second << '=' << bigram.second << '\n';
	}
}

TEST_F(MakemoreTest, Print_Characters)
{
	for (auto c : mm.Characters())
	{
		std::cout << c << '\n';
	}

	for (auto c : mm.Stoi())
	{
		std::cout << '{' << c.first << ',' << c.second << '}' << '\n';
	}

	EXPECT_EQ(mm.Stoi()['a'], 0);
	EXPECT_EQ(mm.Stoi()['@'], 26);
	EXPECT_EQ(mm.Stoi()['$'], 27);
}