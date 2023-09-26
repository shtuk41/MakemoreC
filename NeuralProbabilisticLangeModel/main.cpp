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
	torch::Tensor Y = torch::zeros({ static_cast<long>(label_vector.size()), 1 }, torch::kInt64);

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

	torch::Tensor C = torch::randn({27, 2}, torch::kFloat32);
	//std::cout << C << std::endl;
	torch::Tensor F = torch::one_hot(torch::tensor(5), 27).to(torch::kFloat32);
	std::cout << F.t().unsqueeze(0).mm(C) << std::endl;

	torch::Tensor Embeddings = torch::zeros({ static_cast<long>(context_vector.size()), blockSize, C.size(1)}, torch::kFloat32);

	rowCount = 0;
	for (auto cv : context_vector)
	{
		int colCount = 0;

		for (auto c : cv)
		{
			X[rowCount][colCount] = mm.Stoi()[c];
			Embeddings[rowCount][colCount] = C[mm.Stoi()[c]];
			colCount++;
		}

		rowCount++;
	}
	
	std::cout << X[13][2] << std::endl;
	std::cout << C[1] << std::endl;
	std::cout << Embeddings[13][2] << std::endl;

	torch::Tensor W1 = torch::randn({ 6, 100 });
	torch::Tensor b = torch::randn({ 100 });



	auto t = torch::cat({ Embeddings.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}),
						Embeddings.index({ torch::indexing::Slice(), 1, torch::indexing::Slice() }),
						Embeddings.index({ torch::indexing::Slice(), 2, torch::indexing::Slice() }) }, 1);

	std::cout << "t sizes" << t.sizes() << std::endl;
	std::cout << t << std::endl;

	return 0;
}