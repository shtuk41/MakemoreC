#include "pch.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>
#include <fstream>

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
	double log_likeihood = 0.0;
	double couunt = 0.0;

	for (int ii = 0; ii < mm.Names().size(); ii++)
	{
		auto name = mm.Names()[ii];

		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];

			double probability = mm.Probability(ix1, ix2);
			double logProb = log(probability);

			log_likeihood += logProb;
			couunt += 1.0;

			//std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << "  " << logProb << std::endl;
		}
	}

	std::cout << "log_likelyHood: " << log_likeihood << std::endl;
	std::cout << "negative log_likelyHood: " << -log_likeihood << std::endl;

	double nll = -log_likeihood / couunt;

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
		double log_likeihood = 0.0;
		double couunt = 0.0;

		for (int jj = 0; jj < name.length() - 1; jj++)
		{
			char ch1 = name[jj];
			char ch2 = name[jj + 1];
			int ix1 = mm.Stoi()[ch1];
			int ix2 = mm.Stoi()[ch2];

			double probability = mm.Probability(ix1, ix2);
			double logProb = log(probability);

			log_likeihood += logProb;
			couunt += 1.0;

			std::cout << ch1 << ch2 << ", " << std::setprecision(3) << probability << "  " << logProb << std::endl;
		}

		std::cout << name << '\n';
		std::cout << "log_likelyHood: " << log_likeihood << std::endl;
		std::cout << "negative log_likelyHood: " << -log_likeihood << std::endl;
		double nnl = -log_likeihood / couunt;
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

template <int numberOfElements>
void make_input_x(std::vector<std::shared_ptr<value>>& x, const std::array<double, numberOfElements> &arr)
{
	x.clear();

	for (int ii = 0; ii < numberOfElements; ii++)
	{
		x.push_back(std::make_shared<value>(value(arr[ii], std::string("input") + std::to_string(ii))));
	}
}

void make_input(std::vector<std::shared_ptr<value>>& x, int num, const std::array<double, 27>& arr)
{
	x.clear();

	for (int ii = 0; ii < num; ii++)
	{
		x.push_back(std::make_shared<value>(value(arr[ii], std::string("input") + std::to_string(ii))));
	}
}

TEST_F(MakemoreTest, DISABLED_Network)
{
	std::ofstream monitor("monitor_network.csv");

	monitor << "\n";

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
	layersizes.push_back(5);

	mlp m(5, layersizes);

	std::vector<std::shared_ptr<value>> input_values[numberOfBigrams];
	std::vector<std::shared_ptr<value>> results[numberOfBigrams];

	std::array<double, 5> inputs[numberOfBigrams];

	int count = 0;

	for (auto ii : xs)
	{
		inputs[count] = one_hot<5>(ii);
		count++;
	}
	
	std::vector<std::shared_ptr<value>> values;

	std::vector<std::vector<std::shared_ptr<value>>> probs;
	

	std::vector<std::shared_ptr<value>> likelyhoods;

	for (int pass = 0; pass < 40; pass++)
	{
		
		int labelCount = 0;

		values.clear();
		
		likelyhoods.clear();
		probs.clear();

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			make_input_x<5>(input_values[ii], inputs[ii]);
			results[ii] = m(input_values[ii]);

#ifdef _PRINT
			std::cout << "result " << ii << std::endl;
#endif

			std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
			values.push_back(localSum);
			value::all_values.push_back(localSum);
			labelCount++;

			for (auto jj : results[ii])
			{
#ifdef _PRINT
				std::cout << *jj << ',';
#endif
				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				values.push_back(localSum);
				value::all_values.push_back(localSum);
				labelCount++;
			}

			//if (ii == 1)
			//	trace(*localSum);



#ifdef _PRINT
			std::cout << '\n';

			std::cout << "LocalSum: " << *localSum << std::endl;


			std::cout << '[';
#endif
			std::vector<std::shared_ptr<value>> localProbs;

			for (auto jj : results[ii])
			{
				auto prob = std::make_shared<value>(value(*jj / *localSum)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				value::all_values.push_back(prob);
				labelCount++;

				//if (ii == 1)
				//	trace(*prob);

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
		value::all_values.push_back(oneNeg);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[ii][ys[ii]]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			values.push_back(likelyhood);
			value::all_values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;

			likelyhoods.push_back(likelyhoodNeg);
			value::all_values.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		values.push_back(totalLoss);
		value::all_values.push_back(totalLoss);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[ii])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			values.push_back(totalLoss);
			value::all_values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>((double)numberOfBigrams, std::string("totalNumberOfLosses"));
		value::all_values.push_back(totalNumberOfLosses);

		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss / *totalNumberOfLosses)); loss->set_label(std::string("loss"));
		value::all_values.push_back(loss);

		//trace(*loss);

		//loss->set_grad(1.0);

		loss->backward();

		//trace(*loss);

		


		std::cout << "pass: " << pass << " , LOSS IS: " << *loss << std::endl;

		

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		static bool first = true;

		if (first)
		{
			monitor << ",";
			for (auto ii : params)
			{
				//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

				monitor << ii->label() << "," << ii->label() + "_grad" << ",";
			}

			monitor << "\n";

			first = false;
		}

		monitor << std::to_string(*loss) << ",";

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

			monitor << std::to_string(*ii) << "," << std::to_string(ii->grad()) << ",";

			ii->setData(*ii - 0.1f * ii->grad());
			ii->set_grad(0.0f);
		}

		monitor << "\n";

		value::all_values.clear();
		value::all_weights.clear();

		//if (pass > 0)
		//	trace(*loss);
	}
}

TEST_F(MakemoreTest, Backpropogation1)
{
	std::vector<int> layersizes;
	layersizes.push_back(2);

	mlp m(2, layersizes, 1.0f);

	auto parameters = m.parameters();

	std::cout << "size parameters: " << parameters.size() << std::endl;

	parameters[0]->setData(5.0);
	parameters[1]->setData(-2.5);
	parameters[2]->setData(-0.5);

	std::vector<std::shared_ptr<value>> input_values[2];
	std::vector<std::shared_ptr<value>> results[2];

	std::array<double, 2> inputs[2];

	inputs[0] = one_hot<2>(0);
	inputs[1] = one_hot<2>(1);

	std::vector<std::shared_ptr<value>> values;
	std::vector<std::vector<std::shared_ptr<value>>> probs;
	std::vector<std::shared_ptr<value>> likelyhoods;

	for (int pass = 0; pass < 1; pass++)
	{
		int labelCount = 0;

		for (int bigram = 0; bigram < 2; bigram++)
		{
			make_input_x<2>(input_values[bigram], inputs[bigram]);
			results[bigram] = m(input_values[bigram]);

			std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
			values.push_back(localSum);
			value::all_values.push_back(localSum);
			labelCount++;

			for (auto jj : results[bigram])
			{
				std::cout << *jj << ',';
				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				values.push_back(localSum);
				value::all_values.push_back(localSum);
				labelCount++;
			}

			std::cout << '\n';

			std::vector<std::shared_ptr<value>> localProbs;

			for (auto jj : results[bigram])
			{
				auto prob = std::make_shared<value>(value(*jj / *localSum)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				value::all_values.push_back(prob);
				labelCount++;

				std::cout << *prob << ',';
			}
			std::cout << '\n';

			probs.push_back(localProbs);
		}

		std::shared_ptr<value> oneNeg = std::make_shared<value>(-1.0f, std::string("negone") + std::to_string(labelCount));
		values.push_back(oneNeg);
		value::all_values.push_back(oneNeg);
		labelCount++;

		for (int bigram = 0; bigram < 2; bigram++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[bigram][bigram]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			values.push_back(likelyhood);
			value::all_values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;

			likelyhoods.push_back(likelyhoodNeg);
			value::all_values.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		values.push_back(totalLoss);
		value::all_values.push_back(totalLoss);
		labelCount++;

		for (int bigram = 0; bigram < 2; bigram++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[bigram])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			values.push_back(totalLoss);
			value::all_values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>(2, std::string("totalNumberOfLosses"));
		value::all_values.push_back(totalNumberOfLosses);

		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss / *totalNumberOfLosses)); loss->set_label(std::string("loss"));
		value::all_values.push_back(loss);



		loss->backward();

		trace(*loss);
		trace_split();

		std::cout << "LOSS IS: " << *loss << std::endl;

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		for (auto ii : params)
		{
			std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

			ii->setData(*ii - 0.1f * ii->grad());
			ii->set_grad(0.0f);

			std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';
		}

		values.clear();
		probs.clear();
		likelyhoods.clear();
	}
}

TEST_F(MakemoreTest, DISABLED_Network_ALL)
{
	std::ofstream monitor;

	monitor.open("monitor_network_all.csv");

	ASSERT_TRUE(monitor.is_open());

	
	
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

	std::vector<std::shared_ptr<value>> input_values[numberOfBigrams];
	std::vector<std::shared_ptr<value>> results[numberOfBigrams];

	std::array<double, 27> inputs[numberOfBigrams];

	int count = 0;

	for (auto ii : xs)
	{
		inputs[count] = one_hot<27>(ii);
		count++;
	}

	std::vector<std::vector<std::shared_ptr<value>>> probs;
	std::vector<std::shared_ptr<value>> likelyhoods;
	

	for (int pass = 0; pass < 200; pass++)
	{
		int labelCount = 0;

		for (auto& p : probs)
		{
			p.clear();
		}

		probs.clear();
		likelyhoods.clear();
		m.clear();

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			make_input(input_values[ii], 27, inputs[ii]);
			results[ii] = m(input_values[ii]);

#ifdef _PRINT
			std::cout << "result " << ii << std::endl;
#endif
			std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
			value::all_values.push_back(localSum);
			labelCount++;


			for (auto jj : results[ii])
			{
#ifdef _PRINT
				std::cout << *jj << ',';
#endif

				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				value::all_values.push_back(localSum);
				labelCount++;
			}

#ifdef _PRINT
			std::cout << '\n';

			std::cout << "LocalSum: " << *localSum << std::endl;

			std::cout << '[';
#endif

			std::vector<std::shared_ptr<value>> localProbs;

			for (auto jj : results[ii])
			{
				auto prob = std::make_shared<value>(value(*jj / *localSum)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				value::all_values.push_back(prob);
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
		value::all_values.push_back(oneNeg);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[ii][ys[ii]]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			value::all_values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;
			value::all_values.push_back(likelyhoodNeg);

			likelyhoods.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		value::all_values.push_back(totalLoss);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[ii])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			value::all_values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>((double)numberOfBigrams, std::string("totalNumberOfLosses"));
		value::all_values.push_back(totalNumberOfLosses);

		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss / *totalNumberOfLosses)); loss->set_label(std::string("loss"));
		value::all_values.push_back(loss);

		//trace(*loss);

		loss->set_grad(1.0);
		loss->backward();

		//trace(*loss);

		std::cout << "pass: " << pass << " , LOSS IS: " << *loss << std::endl;

		//monitor << *loss << ',';

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		static bool first = true;

		if (first)
		{
			for (auto ii : params)
			{
				monitor << ii->label() << ",";
			}

			monitor << "\n";

			first = false;
		}

		

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

			//monitor << std::to_string(*ii) << "," << std::to_string(ii->grad()) << ",";

			double grad = ii->grad();
			double value = *ii;
			double newvalue = *ii - 500.0 * ii->grad();

			ii->setData(newvalue);
			ii->set_grad(0.0f);

			monitor << std::to_string(*ii) << ",";
		}

		monitor << "\n";


		value::all_values.clear();
		value::all_weights.clear();
	}

	//monitor.close();

	std::array<double, 27> input;
	
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

			std::vector<double> resultsdouble;

			for (auto r : result)
			{
				resultsdouble.push_back(*r);
			}

			std::discrete_distribution<int> d(resultsdouble.begin(), resultsdouble.end());

			int x = d(gen);

			if (x == 0)
				break;

			name += mm.Itos(x);

			input = one_hot<27>(x);
		}

		std::cout << name << std::endl;
	}
}

TEST_F(MakemoreTest, EXP_Network_ALL)
{
	std::ofstream monitor;

	monitor.open("monitor_network_all.csv");

	ASSERT_TRUE(monitor.is_open());

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

	//m.loadParameters("modelIteration100.csv", std::nullopt);

	auto parameters = m.parameters();

	std::cout << "size parameters: " << parameters.size() << std::endl;

	for (auto ii : parameters)
	{
		monitor << ii->label() << "," << ii->label() + "_graident" << ",";
	}

	monitor << "\n";

	for (auto ii : parameters)
	{
		//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';

		//monitor << std::to_string(*ii) << "," << std::to_string(ii->grad()) << ",";

		double grad = ii->grad();
		ii->set_grad(0.0f);

		monitor << std::to_string(*ii) << "," << std::to_string(grad) << ",";
	}

	monitor << "\n";

	std::vector<std::shared_ptr<value>> *input_values = new std::vector<std::shared_ptr<value>>[numberOfBigrams];
	std::vector<std::shared_ptr<value>> *results = new std::vector<std::shared_ptr<value>>[numberOfBigrams]; 
	std::vector<std::shared_ptr<value>> *interResults = new std::vector<std::shared_ptr<value>>[numberOfBigrams];

	std::array<double, 27> *inputs = new std::array<double, 27>[numberOfBigrams];

	int count = 0;

	for (auto ii : xs)
	{
		inputs[count] = one_hot<27>(ii);
		count++;
	}

	std::vector<std::vector<std::shared_ptr<value>>> probs;
	std::vector<std::shared_ptr<value>> likelyhoods;


	for (int pass = 0; pass < 100; pass++)
	{
		int labelCount = 0;

		for (auto& p : probs)
		{
			p.clear();
		}

		probs.clear();
		likelyhoods.clear();
		m.clear();

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			results[ii].clear();

			make_input(input_values[ii], 27, inputs[ii]);
			interResults[ii] = m(input_values[ii]);

			auto numberIterations = std::make_shared<value>(mm.BigramVector()[ii].second, std::string("numberIterations") + std::to_string(labelCount++));
			value::all_values.push_back(numberIterations);


			for (auto ir : interResults[ii])
			{
#ifdef _PRINT
				std::cout << *ir << ',';
#endif
				auto res = std::make_shared<value>(value(*ir * *numberIterations)); res->set_label(std::string("result") + std::to_string(labelCount++));
				value::all_values.push_back(res);
				results[ii].push_back(res);
			}

#ifdef _PRINT
			std::cout << "result " << ii << std::endl;
#endif
			std::shared_ptr<value> localSum = std::make_shared<value>(0.0f, std::string("localSum") + std::to_string(labelCount));
			value::all_values.push_back(localSum);
			labelCount++;


			for (auto jj : results[ii])
			{
#ifdef _PRINT
				std::cout << *jj << ',';
#endif

				localSum = std::make_shared<value>(value(*localSum + *jj)); localSum->set_label(std::string("localSum") + std::to_string(labelCount));
				value::all_values.push_back(localSum);
				labelCount++;
			}

	
#ifdef _PRINT
			std::cout << '\n';

			std::cout << "LocalSum: " << *localSum << std::endl;

			std::cout << '[';
#endif

			std::vector<std::shared_ptr<value>> localProbs;

			double accumulatedProbabilityCheck = 0.0;

			std::shared_ptr<value> negOne = std::make_shared<value>(-1.0f, std::string("negone") + std::to_string(labelCount));
			value::all_values.push_back(negOne);
			labelCount++;

			for (auto jj : results[ii])
			{
				auto multiplier = std::make_shared<value>(value(localSum->pow(*negOne))); multiplier->set_label(std::string("multiplier") + std::to_string(labelCount));
				value::all_values.push_back(multiplier);
				labelCount++;

				auto prob = std::make_shared<value>(value(*jj * *multiplier)); prob->set_label(std::string("localProb") + std::to_string(labelCount));
				localProbs.push_back(prob);
				value::all_values.push_back(prob);
				labelCount++;

				

#ifdef _PRINT
				accumulatedProbabilityCheck += *prob;
				std::cout << *prob << ',';
#endif
			}

			
#ifdef _PRINT
			std::cout << "]\n";
			std::cout << "accumulated probability: " << accumulatedProbabilityCheck << std::endl;
#endif

			probs.push_back(localProbs);
		}

		traceProbability(std::string("probability") + std::to_string(pass) + ".csv", probs);

		


		std::shared_ptr<value> oneNeg = std::make_shared<value>(-1.0f, std::string("negone") + std::to_string(labelCount));
		value::all_values.push_back(oneNeg);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			auto likelyhood = std::make_shared<value>(value(probs[ii][ys[ii]]->log())); likelyhood->set_label(std::string("likelyhood") + std::to_string(labelCount));
			labelCount++;
			value::all_values.push_back(likelyhood);

			auto likelyhoodNeg = std::make_shared<value>(value(*oneNeg * (*likelyhood))); likelyhoodNeg->set_label(std::string("likelyhoodNeg") + std::to_string(labelCount));
			labelCount++;
			value::all_values.push_back(likelyhoodNeg);

			likelyhoods.push_back(likelyhoodNeg);
		}

		std::shared_ptr<value> totalLoss = std::make_shared<value>(0.0f, std::string("totalLoss") + std::to_string(labelCount));
		value::all_values.push_back(totalLoss);
		labelCount++;

		for (int ii = 0; ii < numberOfBigrams; ii++)
		{
			totalLoss = std::make_shared<value>(value(*totalLoss + *likelyhoods[ii])); totalLoss->set_label(std::string("totalLoss") + std::to_string(labelCount));
			value::all_values.push_back(totalLoss);
			labelCount++;
		}

		std::shared_ptr<value> totalNumberOfLosses = std::make_shared<value>((double)numberOfBigrams, std::string("totalNumberOfLosses") + std::to_string(labelCount));
		value::all_values.push_back(totalNumberOfLosses);
		labelCount++;

		totalNumberOfLosses = std::make_shared<value>(totalNumberOfLosses->pow(*oneNeg), std::string("totalNumberOfLosses") + std::to_string(labelCount));
		value::all_values.push_back(totalNumberOfLosses);
		labelCount++;


		std::shared_ptr<value> loss = std::make_shared<value>(value(*totalLoss * *totalNumberOfLosses)); loss->set_label(std::string("loss") + std::to_string(labelCount));
		value::all_values.push_back(loss);
		labelCount++;

		auto parameters = m.parameters();

		std::shared_ptr<value> totalWeight = std::make_shared<value>(0.0f, std::string("totalWeight") + std::to_string(labelCount));
		value::all_values.push_back(totalWeight);

		for (auto p : parameters)
		{
			totalWeight = std::make_shared<value>(value(*totalWeight + *p)); totalWeight->set_label(std::string("totalWeight") + std::to_string(labelCount));
			value::all_values.push_back(p);
			value::all_values.push_back(totalWeight);
			labelCount++;
		}

		totalWeight = std::make_shared<value>(value(*totalWeight * *totalWeight)); totalWeight->set_label(std::string("totalWeight") + std::to_string(labelCount));
		value::all_values.push_back(totalWeight);
		labelCount++;

		std::shared_ptr<value> numberOfWeights = std::make_shared<value>((double)parameters.size(), std::string("numberOfWeights") + std::to_string(labelCount));
		value::all_values.push_back(numberOfWeights);
		labelCount++;

		numberOfWeights = std::make_shared<value>(numberOfWeights->pow(*oneNeg), std::string("numberOfWeights") + std::to_string(labelCount));
		value::all_values.push_back(numberOfWeights);
		labelCount++;

		totalWeight = std::make_shared<value>(value(*totalWeight * *numberOfWeights)); totalWeight->set_label(std::string("totalWeight") + std::to_string(labelCount));
		value::all_values.push_back(totalWeight);
		labelCount++;

		auto totalWeightScaler = std::make_shared<value>(value(0.01)); totalWeightScaler->set_label(std::string("totalWeightScaler") + std::to_string(labelCount));
		value::all_values.push_back(totalWeightScaler);
		labelCount++;

		totalWeight = std::make_shared<value>(value(*totalWeight * *totalWeightScaler)); totalWeight->set_label(std::string("totalWeight") + std::to_string(labelCount));
		value::all_values.push_back(totalWeight);
		labelCount++;

		std::shared_ptr<value> finalLoss = std::make_shared<value>(value(*loss + *totalWeight)); loss->set_label(std::string("finalLoss"));
		value::all_values.push_back(finalLoss);
		labelCount++;

		//trace(*finalLoss);

		finalLoss->set_grad(1.0);
		finalLoss->backward();

		//trace(*finalLoss);

		std::cout << "pass: " << pass << " , LOSS IS: " << *loss << std::endl;

		//monitor << *loss << ',';

		auto params = m.parameters();

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << ", getSetGrad: " << ii->getSetGradCount() << '\n';

			//monitor << std::to_string(*ii) << "," << std::to_string(ii->grad()) << ",";

			double grad = ii->grad();
			double value = *ii;

			double mult = 50.0;

			double newvalue = *ii - mult * ii->grad();

			ii->setData(newvalue);
			ii->reset_grad();

			monitor << std::to_string(*ii) << "," << std::to_string(grad) << ",";
		}

		monitor << "\n";


		value::all_values.clear();
		value::all_weights.clear();
	}

	//monitor.close();

	std::array<double, 27> input;

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

			std::vector<double> resultsdouble;

			for (auto r : result)
			{
				resultsdouble.push_back(*r);
			}

			std::discrete_distribution<int> d(resultsdouble.begin(), resultsdouble.end());

			int x = d(gen);

			if (x == 0)
				break;

			name += mm.Itos(x);

			input = one_hot<27>(x);
		}

		std::cout << name << std::endl;
	}
}


