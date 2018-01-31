#include "kalman_filter.h"
#include "tools.h"
#include <iostream>
#include <random>
#include <cmath>
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/normal_distribution.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::VectorwiseOp;

using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::MatrixCorrectDimension(const MatrixXd& check, int expectedRows, int expectedCols) {
	if (check.rows() != expectedRows || check.cols() != expectedCols)
		throw WrongDimInputException();
}

void KalmanFilter::VectorCorrectDimension(const VectorXd& check, int expectedSize) {
	if (check.size() != expectedSize)
		throw WrongDimInputException();
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	/*
	VectorCorrectDimension(x_in, 4);
	MatrixCorrectDimension(P_in, 4, 4);
	MatrixCorrectDimension(F_in, 4, 4);
	MatrixCorrectDimension(H_in, 2, 4);
	MatrixCorrectDimension(R_in, 2, 2);
	MatrixCorrectDimension(Q_in, 4, 4);
	*/

	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
 
}


MatrixXd KalmanFilter::GenerateEigenDecomposition(const MatrixXd &Q) {
	
	SelfAdjointEigenSolver<MatrixXd> eigensolver(Q);


	for (int i = 0; i < eigensolver.eigenvalues().size(); i++) {

		if (eigensolver.eigenvalues()(i) <= 0) {
			cout <<"Matrix not positive semidefinite with eigenvalue"<< eigensolver.eigenvalues()(i) << endl;
			throw NotPositiveSemidefinite();
		}
	}

	return eigensolver.eigenvectors()*eigensolver.eigenvalues().cwiseSqrt().asDiagonal();
}

/*
VectorXd KalmanFilter::DrawFromMultivariateGaussianMeanZero(const MatrixXd &Q) {
	static boost::mt19937 rng{ std::random_device{}() };
	static boost::normal_distribution<> dist;
	cout << "eigenvalue decomp is "<< GenerateEigenDecomposition(Q) << endl;
	MatrixCorrectDimension(Q, 4, 4);

	return GenerateEigenDecomposition(Q)*VectorXd { 4 }.unaryExpr([&](auto x) {return dist(rng); });
}
*/



void KalmanFilter::PredictDeterministic() {
	x_ = F_*x_;
	P_ = F_*P_*F_.transpose() + Q_;
}
/*
void KalmanFilter::PredictNonDeterministic() {
	x_ = x_ + DrawFromMultivariateGaussianMeanZero(Q_);
}
*/

void KalmanFilter::Predict() {
	 PredictDeterministic();
	 //PredictNonDeterministic();

}

void KalmanFilter::Update(const VectorXd &z) {


	//VectorXd y(z_size);
	VectorXd y = z - (H_*x_);
	MatrixXd S_ = H_*P_*H_.transpose() + R_;
	MatrixXd K_ = P_*H_.transpose()*S_.inverse();
	x_ = x_ + (K_*y);
	int x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K_*H_)*P_;

}


VectorXd KalmanFilter::CartesianToPolar(double px, double py, double vx, double vy) {
	VectorXd PolarVector(3);
	double r2 = SumSquare(px, py);
	CannotDivideByZero(px);
	CannotDivideByZero(r2);
	double phi = atan2(py , px);
	
	PolarVector << sqrt(r2), phi, (px*vx + py*vy) / sqrt(r2);

	return PolarVector;
}

VectorXd KalmanFilter::TransformPolarToCartesian(const Eigen::VectorXd &PolarVector) {
	VectorXd CartesianVector(4);
	double phi = PolarVector(0);
	double theta = PolarVector(1);
	double phi_dot = PolarVector(2);
	CartesianVector << phi*cos(theta) , phi*sin(theta) , phi_dot*cos(theta) , phi_dot*sin(theta);
	return CartesianVector;
}

double NormalizeRadianBetweenPiMinusPi(double phi) {

	while (phi > M_PI) {
		phi -=  2.0*M_PI;
	}
	
	while (phi < -M_PI) {
		phi +=  2.0*M_PI;
	}

	return phi;

}


VectorXd KalmanFilter::TransformCartesianToPolar(const VectorXd &CartesianVector) {

	InputSizeIs4(CartesianVector);
	CannotDivideByZero(SumSquare(CartesianVector(0), CartesianVector(1)));
	CannotDivideByZero(CartesianVector(0));

	VectorXd PolarVector(3);
	PolarVector = CartesianToPolar(CartesianVector(0), CartesianVector(1), CartesianVector(2), CartesianVector(3));
	return PolarVector;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

	VectorXd y = z - TransformCartesianToPolar(x_);
	y(1) = NormalizeRadianBetweenPiMinusPi(y(1));
	
	
	//Tools tools{ Tools() };
	//MatrixXd Hj = tools.CalculateJacobian(x_);

	MatrixXd S_ = H_*P_*H_.transpose() + R_;
	MatrixXd K_ = P_*H_.transpose() *S_.inverse();

	x_ = x_ + K_*y;
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
	P_ = (I - K_*H_)*P_;
}
