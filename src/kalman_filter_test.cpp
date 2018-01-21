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


