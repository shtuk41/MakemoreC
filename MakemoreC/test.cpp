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
	EXPECT_EQ(mm.N()[1][1], 557);
	EXPECT_EQ(mm.N()[1][2], 542);
	EXPECT_EQ(mm.N()[8][14], 139);
	EXPECT_EQ(mm.N()[0][8], 875);
	EXPECT_EQ(mm.N()[1][0], 6641);
	EXPECT_EQ(mm.N()[14][0], 6764);
}

TEST_F(MakemoreTest, Sample_N)
{
	for (int ii = 0; ii < 20; ii++)
	{
		std::string out = mm.GetNameBySampling();
		std::cout << out << "\n";
	}
}

TEST_F(MakemoreTest, NegativeLogLikelihood)
{
	float log_likeihood = 0.0;
	float couunt = 0.0;

	for (int ii = 0; ii < mm.Names().size(); ii++)
	{
		auto name = mm.Names()[ii];

		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];

			float probability = mm.Probability(ix1, ix2);
			float logProb = log(probability);

			log_likeihood += logProb;
			couunt += 1.0;



			//std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << "  " << logProb << std::endl;
		}
	}

	std::cout << "log_likelyHood: " << log_likeihood << std::endl;
	std::cout << "negative log_likelyHood: " << -log_likeihood << std::endl;

	float nll = -log_likeihood / couunt;

	std::cout << "normalized negative log_likelihood: " << nll << std::endl;

	EXPECT_NEAR(nll, 2.45436, 0.01f);
}

TEST_F(MakemoreTest, LikelihoodKnownNames)
{
	std::vector<std::string> names;

	names.push_back(".andrej.");
	names.push_back(".andrejq.");

	for (int ii = 0; ii < names.size(); ii++)
	{
		auto name = names[ii];
		float log_likeihood = 0.0;
		float couunt = 0.0;

		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];

			float probability = mm.Probability(ix1, ix2);
			float logProb = log(probability);

			log_likeihood += logProb;
			couunt += 1.0;

			//std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << "  " << logProb << std::endl;
		}

		std::cout << name << '\n';
		std::cout << "log_likelyHood: " << log_likeihood << std::endl;
		std::cout << "negative log_likelyHood: " << -log_likeihood << std::endl;
		float nnl = -log_likeihood / couunt;
		std::cout << "normalized negative log_likelihood: " << nnl << std::endl;

		switch (ii)
		{
		case 0: //andrej
			EXPECT_NEAR(nnl, 3.03677f, 0.01f);
			break;
		case 1: //andrejq
			EXPECT_NEAR(nnl, 3.4834f, 0.01f);
			break;
		}

	}
}

TEST_F(MakemoreTest, BigramsTrainingSet)
{
	std::vector<int> xs, ys;


	for (int ii = 0; ii < mm.Names().size(); ii++)
	{
		auto name = mm.Names()[ii];
		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];
			xs.push_back(ix1);
			ys.push_back(ix2);

			std::cout << ch1 << ch2 << std::endl;
		}

		break;
	}
}

