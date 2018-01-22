#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include <exception>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  MatrixXd Jacobian( const double px, const double py, const double vx, const double vy);

};

void InputSizeIs4(const VectorXd& x_state);

double SumSquare(const double px, const double py);

void CannotDivideByZero(const double r2);

void InputHasNoNulls(const VectorXd& x_state);


class WrongDimInputException : public exception {
	virtual const char* what() const throw()
	{
		return "Input vector wrong dimension";
	}
} ;

class ZeroDivideException : public exception {
	virtual const char* what() const throw() {
		return "Attempted to divide by Zero";
	}
};

class NullElementVectorException : public exception {
	virtual const char* what() const throw()
	{
		return "Null element in Vector";
	}
};
template<typename Derived>
bool is_finite(const  Eigen::MatrixBase<Derived>& x);

template<typename Derived>
bool is_nan(const  Eigen::MatrixBase<Derived>& x);


#endif /* TOOLS_H_ */
