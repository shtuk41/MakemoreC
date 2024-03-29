#include <torch/torch.h>
#include <iostream>
#include <string>
#include <Makemore.h>

int main()
{
	Makemore mm;
	mm.Init("..\\SolutionItems\\names.txt", std::nullopt);

	const int hiddenLyerSize = 100;
	const int embeddingSize = 20;

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
		//std::cout << n << std::endl;
		std::string context = start_context;
		std::string full = start_context + n + ".";

		while(true)
		{
			context_vector.push_back(full.substr(0, blockSize));
			label_vector.push_back(full[blockSize]);

			//std::cout << context_vector.back() << " --->" << label_vector.back() << std::endl;

			if (full[blockSize] == '.')
				break;
			full = full.erase(0, 1);
		}
	}

	try
	{
		auto g = torch::Generator();

		torch::Tensor C = torch::randn({ 27, embeddingSize }, torch::kFloat32);
		C.set_requires_grad(true);

		torch::Tensor W1 = torch::randn({ blockSize * embeddingSize, hiddenLyerSize }, torch::kFloat32) * (5/3) / powf(blockSize * embeddingSize, 0.5);
		W1.set_requires_grad(true);
		//torch::Tensor b1 = torch::randn(hiddenLyerSize, torch::kFloat32) * 0.01;
		//b1.set_requires_grad(true);
		torch::Tensor W2 = torch::randn({ hiddenLyerSize,27 }, torch::kFloat32) * 0.1;
		W2.set_requires_grad(true);
		torch::Tensor b2 = torch::randn(27, torch::kFloat32) * 0;
		b2.set_requires_grad(true);

		auto bngain = torch::ones({ 1, hiddenLyerSize });
		bngain.set_requires_grad(true);
		auto bnbias = torch::ones({ 1, hiddenLyerSize });
		bnbias.set_requires_grad(true);

		for (int run = 0; run < 20000; run++)
		{
			auto ix = torch::randint(0, context_vector.size(), { 32, });

			torch::Tensor Embeddings = torch::zeros({ static_cast<long>(ix.size(0)), blockSize, C.size(1) }, torch::kFloat32);
			torch::Tensor Y = torch::zeros({ static_cast<long>(ix.size(0)) }, torch::kInt64);

			int rowCount = 0;
			for (int ii = 0; ii < ix.size(0); ii++)
			{
				auto cv = context_vector[ix[ii].item<int>()];
				int colCount = 0;

				for (auto c : cv)
				{
					Embeddings[rowCount][colCount] = C[mm.Stoi()[c]];
					colCount++;
				}

				auto lv = label_vector[ix[ii].item<int>()];
				Y[rowCount] = mm.Stoi()[lv];

				rowCount++;
			}

			torch::Tensor hpreact = torch::mm(Embeddings.view({ 32, blockSize * embeddingSize }), W1) /* + b1*/;
			//make every neuron guassian
			hpreact = bngain * (hpreact - hpreact.mean(0, true)) / hpreact.std(0, true) + bnbias;
			hpreact = torch::tanh(hpreact);
			torch::Tensor logits = torch::mm(hpreact, W2) + b2;
			//torch::Tensor counts = logits.exp();
			//torch::Tensor prob = counts / counts.sum(1, true);
			//torch::Tensor probsY = torch::zeros({ static_cast<long>(context_vector.size()) }, torch::kFloat32);
			//torch::Tensor loss = -prob.index({ torch::arange((int)context_vector.size(), torch::kInt64), Y }).log().mean();// +0.01 * torch::mm(W, W).mean();
			torch::Tensor loss = torch::nn::functional::cross_entropy(logits, Y, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));

			if (run % 100 == 0)
			{
				std::cout << "Run: " << run << " Loss is : " << loss.item() << std::endl;
			}

			loss.backward();

			float learningRate = 0.1f;

			W1.set_data(W1.data() - learningRate * W1.grad());
			W1.grad().zero_();
			W2.set_data(W2.data() - learningRate * W2.grad());
			W2.grad().zero_();
			//b1.set_data(b1.data() - learningRate * b1.grad());
			//b1.grad().zero_();
			b2.set_data(b2.data() - learningRate * b2.grad());
			b2.grad().zero_();
			C.set_data(C.data() - learningRate * C.grad());
			C.grad().zero_();
			bngain.set_data(bngain.data() - learningRate * bngain.grad());
			bngain.grad().zero_();
			bnbias.set_data(bnbias.data() - learningRate * bnbias.grad());
			bnbias.grad().zero_();
		}

		torch::Tensor Embeddings = torch::zeros({ static_cast<long>(context_vector.size()), blockSize, C.size(1) }, torch::kFloat32);
		torch::Tensor Y = torch::zeros({ static_cast<long>(label_vector.size()) }, torch::kInt64);

		int rowCount = 0;
		for (auto cv : context_vector)
		{
			int colCount = 0;

			for (auto c : cv)
			{
				Embeddings[rowCount][colCount] = C[mm.Stoi()[c]];
				colCount++;
			}

			auto lv = label_vector[rowCount];
			Y[rowCount] = mm.Stoi()[lv];

			rowCount++;
		}

		torch::Tensor hpreact = torch::mm(Embeddings.view({ static_cast<long>(context_vector.size()), blockSize * embeddingSize }), W1) /* + b1*/;
		hpreact = bngain * (hpreact - hpreact.mean(0, true)) / hpreact.std(0, true) + bnbias;
		hpreact = torch::tanh(hpreact);
		torch::Tensor logits = torch::mm(hpreact, W2) + b2;
		torch::Tensor loss = torch::nn::functional::cross_entropy(logits, Y, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));

		std::cout << "Full Loss is: " << loss << std::endl;


		std::string context = start_context;

		int countNames = 0;

		while (countNames < 20)
		{
			std::string name;

			while (true)
			{
				torch::Tensor emb = torch::zeros({ blockSize, C.size(1) }, torch::kFloat32);

				for (int n = 0; n < blockSize; n++)
				{
					emb[n] = C[mm.Stoi()[context[n]]];
				}

				torch::Tensor lh = torch::mm(emb.view({ 1, -1 }), W1) /* + b1*/;
				lh = bngain * (lh - torch::mean(lh)) / torch::std(lh) + bnbias;
				lh = torch::tanh(lh);
				torch::Tensor llogits = torch::mm(lh, W2) + b2;
				torch::Tensor lprobs = torch::softmax(llogits, 1);
				int lix = torch::multinomial(lprobs, 1, true).item<int>();
				context += mm.Itos()[lix];
				context = context.erase(0, 1);
				name += mm.Itos()[lix];

				if (lix == 0)
					break;
			}

			if (name.size() > 1)
			{
				std::cout << name << std::endl;
				countNames++;
			}
		}

		Embeddings.reset();
		Y.reset();
		hpreact.reset();
		logits.reset();
		loss.reset();

		
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