#include <iostream>
#include "tools.h"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace Eigen;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	if (estimations.size() == 0 || ground_truth.size()==0)
		throw WrongDimInputException();

	VectorXd x;
	return x;
}

MatrixXd Tools::Jacobian( const double px, const double py, const double vx, const double vy) {
	double r2 = SumSquare(px, py);
	MatrixXd Hj(3, 4);
	Hj << px / pow(r2, 0.5), py / pow(r2, 0.5), 0, 0,
		-py / r2, px / r2, 0, 0,
		py*(vx*py - vy*px) / pow(r2, 1.5), px*(vy*px - vx*py) / pow(r2, 1.5), px / pow(r2, 0.5), py / pow(r2, 0.5);
	return Hj;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	
	InputSizeIs4(x_state);
	CannotDivideByZero(SumSquare(x_state(0), x_state(1)));
	InputHasNoNulls(x_state);
	
	return Jacobian(x_state(0), x_state(1), x_state(2), x_state(3));
}

// bare implementation directly working on passed expression
// inefficient if the expression x is costly to evaluate, as it will
// be evaluated 4 times
//
// based on these facts: isnan(x) is just x==x
// and is_inf_or_nan(x) is just isnan(x-x)
// code taken from https://forum.kde.org/viewtopic.php?f=74&t=91514
template<typename Derived>
inline bool is_finite(const  Eigen::MatrixBase<Derived>& x)
{
	return ((x - x).array() == (x - x).array()).all();
}

template<typename Derived>
inline bool is_nan(const  Eigen::MatrixBase<Derived>& x)
{
	//This appears to work because nan==nan is false
	return !((x.array() == x.array())).all();
}

void InputHasNoNulls(const VectorXd& x_state) {
	if (is_nan(x_state))
		throw NullElementVectorException();

}

void CannotDivideByZero(const double r2) {
	if (r2 == 0)
		throw ZeroDivideException();
}


void InputSizeIs4(const VectorXd& x_state) {
	if (x_state.size() != 4)
		throw WrongDimInputException();
}

double SumSquare(const double px, const double py) {
	return px*px + py*py;
}
