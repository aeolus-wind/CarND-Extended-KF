#include "kalman_filter.h"
#include "gmock/gmock.h"  
#include "tools.h"
#include <exception>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::is_scalar;

using namespace::testing;




class InputVectorsForJacobianTests : public Test {
public:
	Tools tools;
	VectorXd vectorXInput;
	vector<VectorXd> vectorVectorXInput1;
	vector<VectorXd> vectorVectorXInput2;
};



TEST_F(InputVectorsForJacobianTests, JacobianRaisesExceptionForZeroDenominator) {
	vectorXInput.resize(4);
	vectorXInput << 0, 0, 1, 1;
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), ZeroDivideException);
}

TEST_F(InputVectorsForJacobianTests, JacobianExpects4dVectorInput) {
	vectorXInput.resize(3);
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), WrongDimInputException);
}

TEST_F(InputVectorsForJacobianTests, JacobianReturns3x4Matrix) {
	vectorXInput.resize(4);
	
	ASSERT_THAT(tools.CalculateJacobian(vectorXInput).rows(), 3);
	ASSERT_THAT(tools.CalculateJacobian(vectorXInput).cols(), 4);
}


//Throws null element exception

TEST_F(InputVectorsForJacobianTests, JacobianCompletesNonTrivialExample) {
	vectorXInput.resize(4);
	vectorXInput << 1, 2, 0.2, 0.4;
	MatrixXd Hj(3, 4);
	Hj << 0.44721359549995793, 0.894427, 0, 0,
		-0.4, 0.2, 0, 0,
		0, 0, 0.44721359549995793, 0.89442719099991586;
	MatrixXd output(3, 4);
	output = tools.CalculateJacobian(vectorXInput);

	cout << output(0, 0);
	ASSERT_DOUBLE_EQ(output(0, 0), Hj(0, 0));
	ASSERT_DOUBLE_EQ(output(1, 0), Hj(1, 0));
	ASSERT_DOUBLE_EQ(output(2, 3), Hj(2, 3));
}

TEST_F(InputVectorsForJacobianTests, JacobianThrowsOnNullElementInInput) {
	vectorXInput.resize(4);
	vectorXInput << 1, 2, std::numeric_limits<double>::quiet_NaN(), 0.3;

	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), NullElementVectorException);
}



TEST_F(InputVectorsForJacobianTests, JacobianRaisesExceptionOnNull) {
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), WrongDimInputException);
}

TEST_F(InputVectorsForJacobianTests, CalculateRMSERaisesExceptionOnNull) {
	ASSERT_THROW(tools.CalculateRMSE(vectorVectorXInput1, vectorVectorXInput2), WrongDimInputException);
}

class InputVectorsForRMSETests : public Test {
public:
	vector<VectorXd> estimations;
	vector<VectorXd> ground_truth;
	Tools tool;

	void SetUp() {
		//the input list of estimations
		VectorXd e(4);
		e << 1, 1, 0.2, 0.1;
		estimations.push_back(e);
		e << 2, 2, 0.3, 0.2;
		estimations.push_back(e);
		e << 3, 3, 0.4, 0.3;
		estimations.push_back(e);

		//the corresponding list of ground truth values
		VectorXd g(4);
		g << 1.1, 1.1, 0.3, 0.2;
		ground_truth.push_back(g);
		g << 2.1, 2.1, 0.4, 0.3;
		ground_truth.push_back(g);
		g << 3.1, 3.1, 0.5, 0.4;
		ground_truth.push_back(g);
	}
};

TEST_F(InputVectorsForRMSETests, RMSEFunctionPassesBasicExample) {
	VectorXd expected(4);
	expected << 0.1, 0.1, 0.1, 0.1;

	ASSERT_THAT(tool.CalculateRMSE(estimations, ground_truth)(0), DoubleEq(expected(0)));
	ASSERT_THAT(tool.CalculateRMSE(estimations, ground_truth)(1), DoubleEq(expected(1)));
	ASSERT_THAT(tool.CalculateRMSE(estimations, ground_truth)(2), DoubleEq(expected(2)));
}

int main(int argc, char** argv)
{
	testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
