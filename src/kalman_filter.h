#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
      Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void PredictDeterministic();
  void PredictNonDeterministic();

  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z);

  Eigen::MatrixXd GenerateEigenDecomposition(const Eigen::MatrixXd &Q);

  Eigen::VectorXd DrawFromMultivariateGaussianMeanZero(const Eigen::MatrixXd &Q);

  void MatrixCorrectDimension(const Eigen::MatrixXd& check, int expectedRows, int expectedCols);
  void VectorCorrectDimension(const Eigen::VectorXd& check, int expectedSize);

  Eigen::VectorXd TransformCartesianToPolar(const Eigen::VectorXd &CartesianVector);

  Eigen::VectorXd CartesianToPolar(double px, double py, double vx, double vy);

  Eigen::VectorXd TransformPolarToCartesian(const Eigen::VectorXd &PolarVector);
	
  
};



double NormalizeRadianBetweenPiMinusPi( double phi);

class NotPositiveSemidefinite : public std::exception {
	virtual const char* what() const throw() {
		return "Input matrix not positive semidefinite";
	}
};

#endif /* KALMAN_FILTER_H_ */
