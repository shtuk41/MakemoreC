#include "pch.h"
#include <fstream>
#include <iostream>
#include <iomanip>

#include <Makemore.h>

class MakemoreTest : public ::testing::Test 
{
protected:
	Makemore mm;

	void SetUp() override
	{
		mm.Init("names.txt");
	}

	void TearDown() override
	{
	}
};

TEST_F(MakemoreTest, ReadNames)
{
	size_t length = mm.Names().size();

	EXPECT_EQ(length, 32033);
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

TEST_F(MakemoreTest, BigramStatisticsAllWords_Sorted)
{

	EXPECT_EQ(mm.BigramVector()[0].second, 6763);
}

TEST_F(MakemoreTest, DISABLED_PrintAllBigrams_Sorted)
{

	for (auto bigram : mm.BigramVector())
	{
		std::cout << bigram.first.first << "," << bigram.first.second << '=' << bigram.second << '\n';
	}
}

TEST_F(MakemoreTest, DISABLED_Print_Characters)
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

TEST_F(MakemoreTest, Check_N)
{
	EXPECT_EQ(mm.N()[1][1], 556);
	EXPECT_EQ(mm.N()[1][2], 541);
	EXPECT_EQ(mm.N()[8][14], 138);
	EXPECT_EQ(mm.N()[0][8], 874);
	EXPECT_EQ(mm.N()[1][0], 6640);
	EXPECT_EQ(mm.N()[14][0], 6763);
}

TEST_F(MakemoreTest, Sample_N)
{
	for (int ii = 0; ii < 20; ii++)
	{
		std::string out = mm.GetNameBySampling();
		std::cout << out << "\n";
	}
}

TEST_F(MakemoreTest, BigramProbabilitiesCheck)
{
	for (int ii = 0; ii < 3; ii++)
	{
		auto name = mm.Names()[ii];

		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];

			double probability = mm.Probability(ix1, ix2);



			std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << std::endl;
		}



	}
}