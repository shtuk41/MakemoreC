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
	}

	try
	{
		torch::Tensor xenc = torch::zeros({ static_cast<long>(xs.size()), 27 }, torch::kFloat32);
		torch::Tensor yst = torch::zeros({ static_cast<long>(ys.size()), 1 }, torch::kInt64);

		for (int i = 0; i < ys.size(); i++)
		{
			yst[i] = ys[i];
		}

		//create one hot arrays
		for (int i = 0; i < xs.size(); i++) 
		{
			int index = xs[i];
			xenc[i][index] = 1.0;
		}

		//
		//torch::manual_seed(2147483647);
		auto g = torch::Generator();

		//randn creates random with 0 mean and variance 1
		auto W = torch::randn({ 27, 27 }, torch::kFloat32);
		W.set_requires_grad(true);

		for (int step = 0; step < 1500; step++)
		{
			std::cout << "step number: " << step << std::endl;
			auto logits = torch::mm(xenc, W);
			auto counts = logits.exp();
			auto probs = counts / counts.sum(1, true);

			auto loss = -probs.index({ torch::arange((int)xs.size(), torch::kInt64), yst }).log().mean() + 0.01 * torch::mm(W,W).mean();

			std::cout << "Loss is " << loss << std::endl;

			if (loss.item<double>() < 1.5)
				break;

			loss.backward();

			W.set_data(W.data()  - 50.0 * W.grad());

			W.grad().zero_();

			
		}

		for (int nw = 0; nw < 100; nw++)
		{
			int ix = 0;
			torch::Tensor sel = torch::zeros({ 1, 27 }, torch::kFloat32);
			sel[0][ix] = 1.0;

			std::string name = ".";

			while (true)
			{
				auto logits = torch::mm(sel, W);
				auto counts = logits.exp();
				auto probs = counts / counts.sum(1, true);
				auto ix = torch::multinomial(probs, 1, true).item<int>();
				name += (mm.Itos(ix));

				if (ix == 0)
					break;
			}

			std::cout << name << std::endl;
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}