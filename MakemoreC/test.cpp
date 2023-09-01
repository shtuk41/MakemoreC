#include "pch.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>

#include <functional.h>
#include <Makemore.h>
#include <neuron.h>
#include <trace.h>

//#define _PRINT

class MakemoreTest : public ::testing::Test 
{
protected:
	Makemore mm;
	Makemore mm1;

	void SetUp() override
	{
		mm.Init("names.txt", std::nullopt);
		mm1.Init("names.txt", 1);
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

			std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << "  " << logProb << std::endl;
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

TEST_F(MakemoreTest, OneHot)
{
	auto t = one_hot<27>(13);

	for (auto s = t.begin(); s != t.end(); ++s)
	{
		std::cout << *s << ',';
	}

	std::cout << std::endl;

	std::array<int, 5> input{ 1, 2, 3, 4, 5 };

	auto tt = one_hot<5, 27>(input);

	for (auto it : tt)
	{
		for (auto s : it)
		{
			std::cout << s << ',';
		}

		std::cout << std::endl;
	}
}

void make_input(std::vector<std::shared_ptr<value>>& x, int num, const std::array<float, 27> &arr)
{
	x.clear();

	for (int ii = 0; ii < num; ii++)
	{
		x.push_back(std::make_shared<value>(value(arr[ii], std::string("input") + std::to_string(ii))));
	}
}

TEST_F(MakemoreTest, Network)
{
	const int numberOfBigrams = 5;

	std::vector<int> xs, ys;

	int bigramCount = 0;

	for (int ii = 0; ii < mm1.Names().size(); ii++)
	{
		auto name = mm1.Names()[ii];
		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm1.Stoi()[ch1];
			int ix2 = mm1.Stoi()[ch2];
			xs.push_back(ix1);
			ys.push_back(ix2);

			std::cout << ch1 << ch2 << ',' << ix1 << ',' << ix2 << ',' << std::endl;

			bigramCount++;

			if (bigramCount >= numberOfBigrams)
				break;
			
			
		}
	}

	std::vector<int> layersizes;
	layersizes.push_back(27);

	mlp m(27, layersizes);

	std::vector<std::shared_ptr<value>> x;

	std::vector<std::shared_ptr<value>> input_values[numberOfBigrams];
	std::vector<std::shared_ptr<value>> results[numberOfBigrams];

	std::array<float, 27> inputs[numberOfBigrams];

	int count = 0;

	for (auto ii : xs)
	{
		inputs[count] = one_hot<27>(ii);
		count++;
	}

	

	std::vector<std::shared_ptr<value>> values;

	std::vector<std::vector<std::shared_ptr<value>>> probs;
	std::vector<std::shared_ptr<value>> localProbs;

	std::vector<std::shared_ptr<value>> likelyhoods;

	for (int pass = 0; pass < 40; pass++)
	{
		int labelCount = 0;

		values.clear();
		localProbs.clear();
		likelyhoods.clear();
		probs.clear();

		std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
		values.push_back(localSum);
		labelCount++;
		
		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			make_input(input_values[ii], 27, inputs[ii]);
			results[ii] = m(input_values[ii]);

#ifdef _PRINT
			std::cout << "result " << ii << std::endl;
#endif

			for (auto jj : results[ii])
			{
#ifdef _PRINT
				std::cout << *jj << ',';
#endif

				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				values.push_back(localSum);
				labelCount++;
			}

#ifdef _PRINT
			std::cout << '\n';

			std::cout << "LocalSum: " << *localSum << std::endl;

			std::cout << '[';
#endif

			for (auto jj : results[ii])
			{
				auto prob = std::make_shared<value>(value(*jj / *localSum)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				labelCount++;

#ifdef _PRINT
				std::cout << *prob << ',';
#endif
			}

#ifdef _PRINT
			std::cout << "]\n";
#endif

			probs.push_back(localProbs);
		}

		std::shared_ptr<value> oneNeg = std::make_shared<value>(-1.0f, std::string("negone") + std::to_string(labelCount));
		values.push_back(oneNeg);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[ii][ys[ii]]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;

			likelyhoods.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		values.push_back(totalLoss);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[ii])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>((float)numberOfBigrams, std::string("totalNumberOfLosses"));

		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss / *totalNumberOfLosses)); loss->set_label(std::string("loss"));

		//trace(*loss);

		//loss->set_grad(1.0);
		loss->backward();

		//trace(*loss);

		std::cout << "pass: " << pass << " , LOSS IS: " << *loss << std::endl;

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

			ii->setData(*ii - 0.1f * ii->grad());
			ii->set_grad(0.0f);
		}

		//if (pass > 0)
		//	trace(*loss);
	}
}

TEST_F(MakemoreTest, Network_ALL)
{
	const int numberOfBigrams = 627;

	std::vector<int> xs, ys;

	int bigramCount = 0;

	for (auto ii : mm.BigramVector())
	{
		xs.push_back(mm.Stoi()[ii.first.first]);
		ys.push_back(mm.Stoi()[ii.first.second]);
	}

	std::vector<int> layersizes;
	layersizes.push_back(27);

	mlp m(27, layersizes);

	std::vector<std::shared_ptr<value>> x;

	std::vector<std::shared_ptr<value>> input_values[numberOfBigrams];
	std::vector<std::shared_ptr<value>> results[numberOfBigrams];

	std::array<float, 27> inputs[numberOfBigrams];

	int count = 0;

	for (auto ii : xs)
	{
		inputs[count] = one_hot<27>(ii);
		count++;
	}

	std::vector<std::shared_ptr<value>> values;

	std::vector<std::vector<std::shared_ptr<value>>> probs;
	std::vector<std::shared_ptr<value>> localProbs;

	std::vector<std::shared_ptr<value>> likelyhoods;

	for (int pass = 0; pass < 400; pass++)
	{
		int labelCount = 0;

		values.clear();
		localProbs.clear();
		likelyhoods.clear();

		for (auto& p : probs)
		{
			p.clear();
		}

		probs.clear();

		m.clear();

		std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
		values.push_back(localSum);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			make_input(input_values[ii], 27, inputs[ii]);
			results[ii] = m(input_values[ii]);

#ifdef _PRINT
			std::cout << "result " << ii << std::endl;
#endif

			for (auto jj : results[ii])
			{
#ifdef _PRINT
				std::cout << *jj << ',';
#endif

				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				values.push_back(localSum);
				labelCount++;
			}

#ifdef _PRINT
			std::cout << '\n';

			std::cout << "LocalSum: " << *localSum << std::endl;

			std::cout << '[';
#endif

			for (auto jj : results[ii])
			{
				auto prob = std::make_shared<value>(value(*jj / *localSum)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				labelCount++;

#ifdef _PRINT
				std::cout << *prob << ',';
#endif
			}

#ifdef _PRINT
			std::cout << "]\n";
#endif

			probs.push_back(localProbs);
		}

		std::shared_ptr<value> oneNeg = std::make_shared<value>(-1.0f, std::string("negone") + std::to_string(labelCount));
		values.push_back(oneNeg);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[ii][ys[ii]]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;

			likelyhoods.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		values.push_back(totalLoss);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[ii])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>((float)numberOfBigrams, std::string("totalNumberOfLosses"));

		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss / *totalNumberOfLosses)); loss->set_label(std::string("loss"));

		//trace(*loss);

		//loss->set_grad(1.0);
		loss->backward();

		//trace(*loss);

		std::cout << "pass: " << pass << " , LOSS IS: " << *loss << std::endl;

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

			ii->setData(*ii - 1.0f * ii->grad());
			ii->set_grad(0.0f);
		}
	}

	std::array<float, 27> input;
	
	std::random_device rd;
	auto a = rd();
	std::mt19937 gen(a);

	input = one_hot<27>(0);

	for (int ii = 0; ii < 20; ii++)
	{
		std::string name;
		
		while (true)
		{
			std::vector<std::shared_ptr<value>> input_value;
			std::vector<std::shared_ptr<value>> result;

			make_input(input_value, 27, input);
			result = m(input_value);

			std::vector<float> resultsFloat;

			for (auto r : result)
			{
				resultsFloat.push_back(*r);
			}

			std::discrete_distribution<int> d(resultsFloat.begin(), resultsFloat.end());

			int x = d(gen);

			if (x == 0)
				break;

			name += mm.Itos(x);

			input = one_hot<27>(x);
		}

		std::cout << name << std::endl;
	}




	

	


}


