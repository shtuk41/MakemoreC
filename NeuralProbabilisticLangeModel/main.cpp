#include <torch/torch.h>
#include <iostream>
#include <string>
#include <Makemore.h>

int main()
{
	Makemore mm;
	mm.Init("..\\SolutionItems\\names.txt", 5);

	//context length: how many character do we take to prdict the next one
	int blockSize = 3;
	
	std::string start_context;

	for (int ii = 0; ii < blockSize; ii++)
	{
		start_context += '.';
	}

	std::vector<std::string> context_vector;
	std::vector<char> label_vector;

	for (auto n : mm.OriginalNames())
	{
		std::cout << n << std::endl;
		std::string context = start_context;
		std::string full = start_context + n + ".";

		while(true)
		{
			context_vector.push_back(full.substr(0, blockSize));
			label_vector.push_back(full[blockSize]);

			std::cout << context_vector.back() << " --->" << label_vector.back() << std::endl;

			if (full[blockSize] == '.')
				break;
			full = full.erase(0, 1);
		}
	}

	torch::Tensor X = torch::zeros({ static_cast<long>(context_vector.size()), blockSize }, torch::kFloat32);
	torch::Tensor Y = torch::zeros({ static_cast<long>(label_vector.size())}, torch::kInt64);

	int rowCount = 0;
	for (auto lv : label_vector)
	{
		Y[rowCount] = mm.Stoi()[lv];
		rowCount++;
	}

	rowCount = 0;
	for (auto cv : context_vector)
	{
		int colCount = 0;

		for (auto c : cv)
		{
			X[rowCount][colCount] = mm.Stoi()[c];
			colCount++;
		}

		rowCount++;
	}

	auto g = torch::Generator();
	
	torch::Tensor C = torch::randn({27, 2}, torch::kFloat32);
	C.set_requires_grad(true);
	
	torch::Tensor W1 = torch::randn({ 6, 100 }, torch::kFloat32); 
	W1.set_requires_grad(true);
	torch::Tensor b1 = torch::randn(100, torch::kFloat32); 
	b1.set_requires_grad(true);
	torch::Tensor W2 = torch::randn({ 100,27 }, torch::kFloat32);
	W2.set_requires_grad(true);
	torch::Tensor b2 = torch::randn(27, torch::kFloat32);
	b2.set_requires_grad(true);

	//Embeddings.set_requires_grad(true);

	try
	{

		for (int run = 0; run < 10; run++)
		{
			torch::Tensor Embeddings = torch::zeros({ static_cast<long>(context_vector.size()), blockSize, C.size(1) }, torch::kFloat32);

			rowCount = 0;
			for (auto cv : context_vector)
			{
				int colCount = 0;

				for (auto c : cv)
				{
					torch::Tensor s = C[mm.Stoi()[c]];
					Embeddings[rowCount][colCount] = s;
					colCount++;
				}

				rowCount++;
			}

			torch::Tensor h = torch::tanh(torch::mm(Embeddings.view({ 32,6 }), W1) + b1);
			torch::Tensor logits = torch::mm(h, W2) + b2;
			//torch::Tensor counts = logits.exp();
			//torch::Tensor prob = counts / counts.sum(1, true);
			//torch::Tensor probsY = torch::zeros({ static_cast<long>(context_vector.size()) }, torch::kFloat32);
			//torch::Tensor loss = -prob.index({ torch::arange((int)context_vector.size(), torch::kInt64), Y }).log().mean();// +0.01 * torch::mm(W, W).mean();
			torch::Tensor loss = torch::nn::functional::cross_entropy(logits, Y, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));

			std::cout << "Loss is: " << loss << std::endl;

			loss.backward();

			W1.set_data(W1.data() - 0.1 * W1.grad());
			W1.grad().zero_();
			W2.set_data(W2.data() - 0.1 * W2.grad());
			W2.grad().zero_();
			b1.set_data(b1.data() - 0.1 * b1.grad());
			b1.grad().zero_();
			b2.set_data(b2.data() - 0.1 * b2.grad());
			b2.grad().zero_();
			C.set_data(C.data() - 0.1 * C.grad());
			C.grad().zero_();

		}
	}
	catch (const c10::Error &e)
	{
		std::cout << e.msg() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}




	return 0;
}