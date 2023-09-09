#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <fstream>


#ifndef M_E
#define M_E 2.71828182845904523536f
#endif



class value
{
private:
	double data;
	double _grad;
	std::vector<value*> _prev;
	std::string _op;
	std::string _label;
	static unsigned int numberOfOperations;
	

public:
	value(double d, std::string _label = std::string(""), std::vector<value*> v = std::vector<value*>(), const char* _op = "") : data(d), _op(_op), _label(_label), _grad(0.0f)
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

	const double grad() const
	{
		return _grad;
	}

	void set_grad(double g)
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

		std::string label = std::string("add") + this->label() + other.label();

		value out = value(this->data + other.data, label, nv, "+");

		numberOfOperations++;

		return out;
	}

	value operator-(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		std::string label = std::string("sub") + this->label() + other.label();

		value out = value(this->data - other.data, label, nv, "-");

		numberOfOperations++;

		return out;
	}

	value operator+(double other)
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

		std::string label = std::string("mult") +  this->label() + other.label();

		value out = value(this->data * other.data, label, nv, "*");

		numberOfOperations++;

		return out;
	}

	value operator/(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		std::string label = std::string("div") + this->label() + other.label();

		value out = value(this->data / other.data, label, nv, "/");

		numberOfOperations++;

		return out;
	}

	void setData(double d)
	{
		data = d;
	}

	operator double() const
	{
		return data;
	}

	value tanh()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		std::string label = std::string("tanh") + this->label();

		double t = (double)(std::pow(M_E, 2.0f * data) - 1.0f) / (std::pow(M_E, 2.0f * data) + 1.0f);
		auto out = value(t, label, nv, "tanh");

		numberOfOperations++;

		return out;
	}

	value exp()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		std::string label = std::string("exp") + this->label();

		double t = std::pow(M_E, data);
		auto out = value(t, label, nv, "exp");

		numberOfOperations++;

		return out;
	}

	value pow(value& other)
	{
		std::vector<value*> nv;

		nv.push_back(this);
		nv.push_back(&other);

		std::string label = std::string("pow") + this->label();

		double t = std::pow(data, other);
		auto out = value(t, label, nv, "pow");

		numberOfOperations++;

		return out;
	}

	value log()
	{
		std::vector<value*> nv;

		nv.push_back(this);

		std::string label = std::string("log") + this->label();

		double t = std::log(data);
		auto out = value(t, label, nv, "log");

		numberOfOperations++;

		return out;
	}

	void calc_backward()
	{
#ifdef _DEBUG_FILE
		std::ofstream traceFile("trace.txt", std::ios::app);
		traceFile << "starting " <<  this->label()  << "\n";
#endif



		if (_op.compare("tanh") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation tanh\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << *_prev[0] << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + (1.0f - std::pow(data, 2.0f)) * grad());

#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
		}
		else if (_op.compare("+") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation +\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + 1.0f * grad());
			_prev[1]->set_grad(_prev[1]->grad() + 1.0f * grad());

#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif
		}
		else if (_op.compare("-") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation -\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + 1.0f * grad());
			_prev[1]->set_grad(_prev[1]->grad() + 1.0f * grad());
#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif
		}
		else if (_op.compare("*") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation *\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + double(*_prev[1]) * grad());
			_prev[1]->set_grad(_prev[1]->grad() + double(*_prev[0]) * grad());

#ifdef _DEBUG_FILE
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif
		}
		else if (_op.compare("exp") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation exp\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + std::pow(M_E, double(*_prev[0])) * grad());

#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
		}
		else if (_op.compare("pow") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation pow\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
			_prev[0]->set_grad(_prev[0]->grad() + double(*_prev[1]) * std::pow(double(*_prev[0]), double(*_prev[1]) - 1.0f) * grad());

#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
		}
		else if (_op.compare("log") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation log\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
			_prev[0]->set_grad(_prev[0]->grad() + 1.0f / double(*_prev[0]) * grad());
#ifdef _DEBUG_FILE
			traceFile << "after change\n";

			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
#endif
		}
		else if (_op.compare("/") == 0)
		{
#ifdef _DEBUG_FILE
			traceFile << "operation /\n";
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif

			_prev[0]->set_grad(_prev[0]->grad() + 1.0f / double(*_prev[1]) * grad());
			_prev[1]->set_grad(_prev[1]->grad() + double(*_prev[0]) / (double(*_prev[1]) + grad()) - double(*_prev[0]) /  double(*_prev[1]));
#ifdef _DEBUG_FILE		
			traceFile << this->label() << " this gradient: " << grad() << "\n";
			traceFile << this->label() << " this data: " << data << "\n";
			traceFile << _prev[0]->label() << " prev[0] gradient: " << _prev[0]->grad() << "\n";
			traceFile << _prev[0]->label() << " prev[0] data: " << double(*_prev[0]) << "\n";
			traceFile << _prev[1]->label() << " prev[0] gradient: " << _prev[1]->grad() << "\n";
			traceFile << _prev[1]->label() << " prev[0] data: " << double(*_prev[1]) << "\n";
#endif
		}
#ifdef _DEBUG_FILE
		traceFile << "end\n";
#endif
		return;
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

	void backward()
	{
		//auto start = std::chrono::high_resolution_clock::now();

		//topo.clear();
		//visited.clear();

		_grad = 1.0f;

		//build_topo(this);
		//auto stop = std::chrono::high_resolution_clock::now();
		//auto topoduration = duration_cast<std::chrono::seconds>(stop - start);
		//std::cout << "Topo discovery took: " << topoduration.count() << " seconds. " << std::endl;

		//std::cout << "TOPO SIZE: " << topo.size() << std::endl;
		
		auto start = std::chrono::high_resolution_clock::now();

		//for (auto it = topo.rbegin(); it != topo.rend(); ++it)
		//{
		//	(*it)->calc_backward();
		//}

		for (auto it = value::all_values.rbegin() ; it != value::all_values.rend(); ++it)
		{
			//std::cout << "backwards: " << (*it)->label() << "\n";
			(*it)->calc_backward();
		}

		for (auto it = value::all_weights.rbegin(); it != value::all_weights.rend(); ++it)
		{
			//std::cout << "backwards weights: " << (*it)->label() << "\n";
			(*it)->calc_backward();
		}

		auto stop = std::chrono::high_resolution_clock::now();

		auto calcduration = duration_cast<std::chrono::microseconds>(stop - start);

		std::cout << "Calc backward took: " << calcduration.count() << " microseconds. " << std::endl;
	}

	static std::vector<std::shared_ptr<value>> all_values;
	static std::vector <std::shared_ptr<value>> all_weights;
};

value operator+(double f, value& v);
value operator*(double f, value& v);

