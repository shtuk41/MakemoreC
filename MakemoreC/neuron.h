#pragma once

#include <assert.h>
#include <vector>
#include <random>
#include <memory>
#include <value.h>
#include <optional>
#include <fstream>


class neuron
{
private:
	size_t numberOfInputs;
	std::vector<std::shared_ptr<value>> weights;
	std::vector<std::shared_ptr<value>> values;
	std::vector< std::vector<std::shared_ptr<value>>> values_mem;
	std::shared_ptr<value> out;
	static int count;
	static int name;

	std::string neuron_name;

public:
	neuron(int nin, std::optional<double> defaultWeight) :numberOfInputs(nin)//, out(std::make_shared<value>(-9999999.9f, "invalid"))
	{
		neuron_name = std::string("_neu_") + std::to_string(name);
		name++;

		if (defaultWeight.has_value())
		{
			for (int ii = 0; ii < nin; ii++)
			{
				auto i = std::make_shared<value>(defaultWeight.value(), std::string("weight") + std::to_string(ii) + neuron_name);
				weights.push_back(i);
			}
		}
		else
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::normal_distribution<double> dis(0.0, 1.0f);

			for (int ii = 0; ii < nin; ii++)
			{
				auto i = std::make_shared<value>((double)dis(gen), std::string("weight") + std::to_string(ii) + neuron_name);
				weights.push_back(i);
			}
		}
	}

	neuron(std::initializer_list<double> weightvals)
	{
		int count = 0;

		for (double val : weightvals)
		{
			weights.push_back(std::make_shared<value>(val, std::string("weight" + std::to_string(count) + neuron_name)));
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

		

		std::string name;

		while (itw != weights.end() && iti != inputs.end())
		{
			value::all_weights.push_back(*itw);
			value::all_weights.push_back(*iti);
			auto mult = std::make_shared<value>(value(*(*itw) * **iti)); mult->set_label(std::string("neu_mult") + std::to_string(count) + neuron_name);
			values.push_back(mult);
			value::all_weights.push_back(mult);
			count++;
			++itw;
			++iti;
		}

		std::shared_ptr<value> zero = std::make_shared<value>(0.0f, std::string("neu_zero") + std::to_string(count) + neuron_name);
		values.push_back(zero);
		value::all_weights.push_back(zero);
		count++;

		for (int ii = 0; ii < numberOfInputs; ii++)
		{
			zero = std::make_shared<value>(value(*zero + *values[ii])); zero->set_label(std::string("neu_add") + std::to_string(count) + neuron_name);
			values.push_back(zero);
			value::all_weights.push_back(zero);
			count++;
		}

		out = std::make_shared<value>(value(zero->exp())); out->set_label(std::string("neu_out") + std::to_string(count) + neuron_name);
		count++;
		value::all_weights.push_back(out);

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

	void clear()
	{
		for (auto& r : values_mem)
		{
			r.clear();
		}

		values_mem.clear();
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
	layer(int nin, int nout, std::optional<double> defaultWeight) : numberOfInputs(nin), numberOfOutputs(nout)
	{
		for (int ii = 0; ii < nout; ii++)
		{
			neurons.push_back(std::make_shared<neuron>(nin, defaultWeight));
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

	void clear()
	{
		for (auto& n : neurons)
		{
			n->clear();
		}

		for (auto &r : outs_mem)
		{
			r.clear();
		}

		std::vector<std::vector<std::shared_ptr<value>>> v;

		outs_mem.clear();
	}
};

class mlp
{
private:
	std::vector<layer> layers;
	std::vector<std::vector<std::shared_ptr<value>>> results;

public:
	mlp(int nin, std::vector<int> nouts, std::optional<double> defaultWeight = std::nullopt)
	{
		std::vector<int> sz;
		sz.push_back(nin);
		sz.insert(sz.end(), nouts.begin(), nouts.end());

		for (int ii = 0; ii < nouts.size(); ii++)
		{
			layers.push_back(layer(sz[ii], sz[ii + 1], defaultWeight));
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

	void clear()
	{
		for (auto &l : layers)
		{
			l.clear();
		}

		for (auto &r : results)
		{
			r.clear();
		}

		results.clear();
	}

	size_t saveParameters(std::string fileName)
	{
		std::ofstream file(fileName);
		auto params = parameters();

		for (auto p : params)
		{
			file << std::to_string(*p) << "," << std::to_string(p->grad()) << ",";
		}

		file.close();

		return params.size() * 2;
	}

	bool loadParameters(std::string fileName, size_t count)
	{
		std::ifstream file(fileName);

		std::string line;
		std::getline(file, line);

		std::vector<std::string> parameterStrings;
		std::string cell;

		std::stringstream lineStream(line);
		while (std::getline(lineStream, cell, ','))
		{
			parameterStrings.push_back(cell);
		}

		if (parameterStrings.size() != count)
		{
			return false;
		}

		auto params = parameters();

		auto iterator = parameterStrings.begin();

		for (auto &p : params)
		{
			double v = std::stod(*iterator++);
			double g = std::stod(*iterator++);

			p->setData(v);
			p->set_grad(g);
		}
	}
};

int neuron::count = 0;
int neuron::name = 0;

