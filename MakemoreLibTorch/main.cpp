#include <torch/torch.h>
#include <iostream>
#include <Makemore.h>

int main()
{
	Makemore mm;
	mm.Init("..\\SolutionItems\\names.txt", std::nullopt);

	std::vector<int> xs, ys;

	for (auto ii : mm.BigramVector())
	{
		xs.push_back(mm.Stoi()[ii.first.first]);
		ys.push_back(mm.Stoi()[ii.first.second]);
		//std::cout << ii.first.first << "," << ii.first.second << "\n";
		//std::cout << xs.back() << "\n";
	}

	try
	{
		torch::Tensor xenc = torch::zeros({ static_cast<long>(xs.size()), 27 }, torch::kFloat32);
		torch::Tensor yst = torch::zeros({ static_cast<long>(ys.size()), 1 }, torch::kInt64);

		for (int i = 0; i < ys.size(); i++)
		{
			yst[i] = ys[i];
		}

		for (int i = 0; i < xs.size(); i++) 
		{
			int index = xs[i];
			xenc[i][index] = 1.0;
		}

		//
		torch::manual_seed(2147483647);
		auto W = torch::randn({ 27, 27 }, torch::kFloat32);
		W.set_requires_grad(true);
		// ...
		W.set_requires_grad(true);
		auto logits = torch::mm(xenc, W);
		auto counts = logits.exp();
		auto probs = counts / counts.sum(1, true);

		auto loss = -probs.index({torch::arange((int)xs.size(), torch::kInt64), yst}).log().mean();


		

		loss.backward();

		std::cout << W.grad() << std::endl;

		W.grad().zero_();

		std::cout << W.grad() << std::endl;
	
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	

}