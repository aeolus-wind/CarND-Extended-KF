#include "kalman_filter.h"
#include "gmock/gmock.h"  
#include "tools.h"
#include <exception>
#include "kalman_filter.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using Eigen::SelfAdjointEigenSolver;

using namespace::testing;


class TestEigenDecompositionMatrix : public Test {
public:
	KalmanFilter kf;
	
	MatrixXd EigenvalueMatrix;

};



TEST_F(TestEigenDecompositionMatrix, EigenSolverReturnsEigenValuesAsMatrix) {
	EigenvalueMatrix.resize(2, 2);
	EigenvalueMatrix << 1, 0, 0, 4;
	SelfAdjointEigenSolver<MatrixXd> Solver(EigenvalueMatrix);
	ASSERT_EQ(Solver.eigenvalues()(0), EigenvalueMatrix(0, 0));
	MatrixXd ExpectedEigenVectors(2, 2);
	ExpectedEigenVectors << 1, 0, 0, 1;

	ASSERT_TRUE(Solver.eigenvectors().isApprox(ExpectedEigenVectors));
}


TEST_F(TestEigenDecompositionMatrix, GenerateEigenDecompositionReturnsOriginalAfterOutputTimeOutput_Transpose) {
	EigenvalueMatrix.resize(2, 2);
	EigenvalueMatrix << 2, -2, -2, 2;
	MatrixXd output(kf.GenerateEigenDecomposition(EigenvalueMatrix));

	cout << output << endl;
	ASSERT_TRUE((output*output.transpose()).isApprox(EigenvalueMatrix));
}




TEST_F(TestEigenDecompositionMatrix, DrawFromMultivariateGaussianMeanZeroReturns4dVector) {
	EigenvalueMatrix.resize(4,4);
	EigenvalueMatrix << 1, 0, 0, 0,
						0, 2, 0, 0,
						0, 0, 3, 0,
						0, 0, 0, 4;
	cout << "eigen decomp is" << kf.GenerateEigenDecomposition(EigenvalueMatrix) << endl;
	ASSERT_EQ(kf.DrawFromMultivariateGaussianMeanZero(EigenvalueMatrix).size(), 4);
}


TEST_F(TestEigenDecompositionMatrix, GenerateEigenDecompositionThrowsExceptionIfInputNotPositiveSemiDefinite) {
	EigenvalueMatrix.resize(2, 2);
	EigenvalueMatrix << 1, 2, 3, 4;

	ASSERT_THROW(kf.GenerateEigenDecomposition(EigenvalueMatrix), NotPositiveSemidefinite);
}



TEST(TestPredictDimInput, DISABLED_GenerateEigenDecompositionThrowsWrongDimInputExceptionIfInputNot4x4) {
	KalmanFilter kf;
	MatrixXd ProcessCovMatrix(2,2);
	ProcessCovMatrix << 1, 0, 0, 4;
	kf.Q_ = ProcessCovMatrix;
	ASSERT_THROW(kf.Predict(), WrongDimInputException);
}


class KalmanFilterInvalidInitializationOnP_ : public Test {
public:
	VectorXd x_in;
	MatrixXd P_in;
	MatrixXd F_in;
	MatrixXd H_in;
	MatrixXd R_in;
	MatrixXd Q_in;
	KalmanFilter kf;

	void SetUp() override {
		x_in.resize(4);
		P_in.resize(4, 3); //incorrect dimensions
		F_in.resize(4, 4);
		H_in.resize(2, 4);
		R_in.resize(2, 2);
		Q_in.resize(4, 4);
	}

};

TEST_F(KalmanFilterInvalidInitializationOnP_, KalmanFilterThrowsInvalidInputExceptionOnIncorrectDimensionInput) {
	ASSERT_THROW(kf.Init(x_in, P_in, F_in, H_in, R_in, Q_in), WrongDimInputException);
}

class KalmanFilterNonTrivialExample : public Test {
public:
	VectorXd x_in;
	MatrixXd P_in;
	MatrixXd F_in;
	MatrixXd H_in;
	MatrixXd R_in;
	MatrixXd Q_in;
	KalmanFilter kf;
	VectorXd firstMeasurement;
	double dt;
	double noise_ax;
	double noise_ay;

	void SetUp() override {
		x_in.resize(4);
		P_in.resize(4, 4);
		F_in.resize(4, 4);
		H_in.resize(2, 4);
		R_in.resize(2, 2);
		Q_in.resize(4, 4);

		x_in << 0.463227, 0.607415, 0, 0;
		P_in << 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1000, 0,
			0, 0, 0, 1000;
		R_in << 0.0225, 0,
			0, 0.0225;
		H_in << 1, 0, 0, 0,
			0, 1, 0, 0;
		F_in << 1, 0, 1, 0,
			0, 1, 0, 1,
			0, 0, 1, 0,
			0, 0, 0, 1;
		noise_ax = 5;
		noise_ay = 5;
		dt = 0.1;
		double dT4 = pow(dt, 4) / 4;
		double dT3 = pow(dt, 3) / 2;
		double dT2 = pow(dt, 2);
		Q_in << dT4*noise_ax, 0, dT3*noise_ax, 0,
			0, dT4*noise_ay, 0, dT3*noise_ay,
			dT3*noise_ax, 0, dT2*noise_ax, 0,
			0, dT3*noise_ay, 0, dT2*noise_ay;

		firstMeasurement.resize(2);
		firstMeasurement << 0.968521, 0.40545;
	}

};

TEST_F(KalmanFilterNonTrivialExample, NoThrowsWithInitializationMatricesAndLength4x_in) {
	ASSERT_NO_THROW(kf.Init(x_in, P_in, F_in, H_in, R_in, Q_in));
}

TEST_F(KalmanFilterNonTrivialExample, NonTrivialExampleReturnsExpectedValueOnPredictDeterministic) {
	kf.Init(x_in, P_in, F_in, H_in, R_in, Q_in);
	VectorXd ExpectedAfterOnePredict(4);
	ExpectedAfterOnePredict << 0.463227, 0.607415, 0, 0;
	kf.PredictDeterministic();
	ASSERT_EQ(kf.x_(0),ExpectedAfterOnePredict(0));
	ASSERT_EQ(kf.x_(1), ExpectedAfterOnePredict(1));
	ASSERT_EQ(kf.x_(2), ExpectedAfterOnePredict(2));
}

TEST_F(KalmanFilterNonTrivialExample, NonTrivialExampleReturnsExpectedValueAfterUpdate) {
	kf.Init(x_in, P_in, F_in, H_in, R_in, Q_in);
	MatrixXd ExpectedP_AfterOneUpdate(4, 4);
	ExpectedP_AfterOneUpdate << 0.022454071740052282, 0, 0.204131, 0,
								0, 0.0224541, 0, 0.204131,
								0.204131, 0, 92.7787, 0,
								0, 0.204131, 0, 92.7787;
	kf.F_(0, 2) = dt;
	kf.F_(1, 3) = dt;
	kf.PredictDeterministic();
	kf.Update(firstMeasurement);

	cout << kf.P_ << "P_ is in its entirety " << endl;

	ASSERT_DOUBLE_EQ(kf.P_(0, 0), ExpectedP_AfterOneUpdate(0, 0));
	ASSERT_DOUBLE_EQ(kf.P_(0, 1), ExpectedP_AfterOneUpdate(0, 1));
}

class PolarTransformExamples : public Test {
public:
	VectorXd SampleVelocity;
	KalmanFilter kf;

};



TEST_F(PolarTransformExamples, TransformCartesianToPolarThrowsIfInputNotSize4) {
	SampleVelocity.resize(3);

	ASSERT_THROW(kf.TransformCartesianToPolar(SampleVelocity), WrongDimInputException);
}

TEST_F(PolarTransformExamples, TransformCartesianToPolarThrowsZeroDivideExceptionOnzeroVelocity) {
	SampleVelocity.resize(4);
	SampleVelocity << 0, 0, 0, 0;
	ASSERT_THROW(kf.TransformCartesianToPolar(SampleVelocity), ZeroDivideException);
}

TEST_F(PolarTransformExamples, TransformCartesianToPolarReturnsOutputBetweenNegativePiandPiInArctanCoord) {
	SampleVelocity.resize(4);
	SampleVelocity << 1, 1, 0, 0;
	//write a real example of this...
	ASSERT_THAT(kf.TransformCartesianToPolar(SampleVelocity)(1), testing::AllOf(testing::Ge(-M_PI), testing::Le(M_PI)));
}