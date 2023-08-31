#pragma once

#include <assert.h>
#include <vector>
#include <random>
#include <memory>
#include <value.h>


class neuron
{
private:
	size_t numberOfInputs;
	std::vector<std::shared_ptr<value>> weights;
	std::vector<std::shared_ptr<value>> values;
	std::vector< std::vector<std::shared_ptr<value>>> values_mem;
	std::shared_ptr<value> out;

public:
	neuron(int nin) :numberOfInputs(nin), out(std::make_shared<value>(-9999999.9f, "invalid"))
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dis(0.0, 1.0f);

		for (int ii = 0; ii < nin; ii++)
		{
			auto i = std::make_shared<value>(dis(gen), std::string("weight") + std::to_string(ii));
			weights.push_back(i);
		}
	}

	neuron(std::initializer_list<float> weightvals, float biasval)
	{
		int count = 0;

		for (float val : weightvals)
		{
			weights.push_back(std::make_shared<value>(val, std::string("weight" + std::to_string(count))));
			count++;
		}

		numberOfInputs = weights.size();
	}

	std::vector<std::shared_ptr<value>> parameters()
	{
		std::vector<std::shared_ptr<value>> out = weights;

		return out;
	}

	std::shared_ptr<value> operator()(const std::vector<std::shared_ptr<value>>& inputs)
	{
		assert(weights.size() == inputs.size());

		values.clear();

		auto itw = weights.begin();
		std::vector<std::shared_ptr<value>>::const_iterator iti = inputs.begin();

		static int count = 0;

		std::string name;

		while (itw != weights.end() && iti != inputs.end())
		{
			auto mult = std::make_shared<value>(value(*(*itw) * **iti)); mult->set_label(std::string("mult") + std::to_string(count));
			values.push_back(mult);
			count++;
			++itw;
			++iti;
		}

		std::shared_ptr<value> zero = std::make_shared<value>(0.0f, std::string("zero") + std::to_string(count));
		values.push_back(zero);
		count++;

		for (int ii = 0; ii < numberOfInputs; ii++)
		{
			zero = std::make_shared<value>(value(*zero + *values[ii])); zero->set_label(std::string("add") + std::to_string(count));
			values.push_back(zero);
			count++;
		}

		auto act = std::make_shared<value>(value(*values.back())); act->set_label(std::string("act") + std::to_string(count));
		count++;
		values.push_back(act);

		out = std::make_shared<value>(value(values.back()->exp())); out->set_label(std::string("out") + std::to_string(count));
		count++;

		values_mem.push_back(values);

		return out;
	}

	void print()
	{
		std::cout << values.size() << std::endl;
		for (auto ii : values)
		{
			std::cout << "label: " << ii->label() << std::endl;
			std::cout << "value: " << *ii << std::endl;
		}
	}

	std::shared_ptr<value> GetOutput() const
	{
		return out;
	}
};

class layer
{
private:
	size_t numberOfInputs;
	size_t numberOfOutputs;
	std::vector<std::shared_ptr<neuron>> neurons;
	std::vector<std::shared_ptr<value>> outs;
	std::vector< std::vector<std::shared_ptr<value>>> outs_mem;


public:
	layer(int nin, int nout) : numberOfInputs(nin), numberOfOutputs(nout)
	{
		for (int ii = 0; ii < nout; ii++)
		{
			neurons.push_back(std::make_shared<neuron>(nin));
		}
	}

	std::vector<std::shared_ptr<value>> parameters()
	{
		std::vector<std::shared_ptr<value>> params;

		for (auto ii : neurons)
		{
			auto p = ii->parameters();
			params.insert(params.end(), p.begin(), p.end());
		}

		return params;
	}

	std::vector<std::shared_ptr<value>> operator()(std::vector<std::shared_ptr<value>> x)
	{
		outs.clear();

		for (std::vector<std::shared_ptr<neuron>>::iterator it = neurons.begin(); it != neurons.end(); ++it)
		{
			outs.push_back((**it)(x));
		}

		outs_mem.push_back(outs);

		return outs_mem.back();
	}
};

class mlp
{
private:
	std::vector<layer> layers;
	std::vector<std::vector<std::shared_ptr<value>>> results;

public:
	mlp(int nin, std::vector<int> nouts)
	{
		std::vector<int> sz;
		sz.push_back(nin);
		sz.insert(sz.end(), nouts.begin(), nouts.end());

		for (int ii = 0; ii < nouts.size(); ii++)
		{
			layers.push_back(layer(sz[ii], sz[ii + 1]));
		}
	}

	std::vector<std::shared_ptr<value>> parameters()
	{
		std::vector<std::shared_ptr<value>> outs;

		for (auto ii : layers)
		{
			auto p = ii.parameters();
			outs.insert(outs.end(), p.begin(), p.end());
		}

		return outs;
	}

	std::vector<std::shared_ptr<value>> operator()(std::vector<std::shared_ptr<value>> x)
	{
		auto a = x;

		for (std::vector<layer>::iterator it = layers.begin(); it != layers.end(); ++it)
		{
			a = (*it)(a);
		}

		results.push_back(a);

		return results.back();
	}
};

