#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#ifndef M_E
#define M_E 2.71828182845904523536f
#endif



class value
{
private:
	float data;
	float _grad;
	std::vector<value*> _prev;
	std::string _op;
	std::string _label;
	static unsigned int numberOfOperations;

public:
	value(float d, std::string _label = std::string(""), std::vector<value*> v = std::vector<value*>(), const char* _op = "") : data(d), _op(_op), _label(_label), _grad(0.0f)
	{
		_prev = v;
	}

	std::string op() const
	{
		return _op;
	}

	std::vector<value*> prev() const
	{
		return _prev;
	}

	const std::string& label() const
	{
		return _label;
	}

	void set_label(std::string l)
	{
		_label = l;
	}

	void set_label(const char* l)
	{
		_label = l;
	}

	const float grad() const
	{
		return _grad;
	}

	void set_grad(float g)
	{
		_grad = g;
	}

	void print() const
	{
		std::cout << "value(data=" << this->data << ")";
	}

	friend std::ostream& operator<<(std::ostream& out, value const& v)
	{
		return std::cout << v.data;
	}

	value operator+(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		value out = value(this->data + other.data, "", nv, "+");

		numberOfOperations++;

		return out;
	}

	value operator-(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		value out = value(this->data - other.data, "", nv, "-");

		numberOfOperations++;

		return out;
	}

	value operator+(float other)
	{
		this->data += other;
		numberOfOperations++;
		return *this;
	}

	value operator*(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		value out = value(this->data * other.data, "", nv, "*");

		numberOfOperations++;

		return out;
	}

	value operator/(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		value out = value(this->data / other.data, "", nv, "/");

		numberOfOperations++;

		return out;
	}

	void setData(float d)
	{
		data = d;
	}

	operator float() const
	{
		return data;
	}

	value tanh()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		float t = (std::pow(M_E, 2.0f * data) - 1.0f) / (std::pow(M_E, 2.0f * data) + 1.0f);
		auto out = value(t, "", nv, "tanh");

		numberOfOperations++;

		return out;
	}

	value exp()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		float t = std::pow(M_E, data);
		auto out = value(t, "", nv, "exp");

		numberOfOperations++;

		return out;
	}

	value pow(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		float t = std::pow(data, other);
		auto out = value(t, "", nv, "pow");

		numberOfOperations++;

		return out;
	}

	value log()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		float t = std::log(data);
		auto out = value(t, "", nv, "log");

		numberOfOperations++;

		return out;
	}

	std::vector<value*> topo;
	std::vector<value*> visited;

	void build_topo(value* v)
	{
		bool visit = false;

		for (auto jj : visited)
		{
			if (jj == v)
			{
				visit = true;
				break;
			}
		}

		if (!visit)
		{
			visited.push_back(v);

			for (auto ii : v->prev())
			{
				build_topo(ii);
			}
			topo.push_back(v);
		}
	}

	void calc_backward()
	{
		if (_op.compare("tanh") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + (1.0f - std::pow(data, 2.0f)) * grad());
		}
		else if (_op.compare("+") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + 1.0f * grad());
			_prev[1]->set_grad(_prev[1]->grad() + 1.0f * grad());
		}
		else if (_op.compare("-") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + 1.0f * grad());
			_prev[1]->set_grad(_prev[1]->grad() + 1.0f * grad());
		}
		else if (_op.compare("*") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + *_prev[1] * grad());
			_prev[1]->set_grad(_prev[1]->grad() + *_prev[0] * grad());
		}
		else if (_op.compare("exp") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + data * grad());
		}
		else if (_op.compare("pow") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + *_prev[1] * std::pow(*_prev[0], *_prev[1] - 1.0f) * grad());
		}
		else if (_op.compare("log") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + 1.0f / *_prev[0] * grad());
		}
		else if (_op.compare("/") == 0)
		{
			_prev[0]->set_grad(_prev[0]->grad() + 1.0f / *_prev[1] * grad());
			_prev[1]->set_grad(_prev[1]->grad() - *_prev[0] / (*_prev[1] * *_prev[1]) * grad());
		}

		return;
	}

	void backward()
	{
		topo.clear();
		visited.clear();

		_grad = 1.0f;

		build_topo(this);

		std::cout << "TOPO SIZE: " << topo.size() << std::endl;

		for (auto it = topo.rbegin(); it != topo.rend(); ++it)
		{
			(*it)->calc_backward();
		}
	}
};

value operator+(float f, value& v);
value operator*(float f, value& v);

