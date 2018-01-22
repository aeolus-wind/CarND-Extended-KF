#include "kalman_filter.h"
#include "gmock/gmock.h"  
#include "tools.h"
#include <exception>
#include <boost/random/mersenne_twister.hpp>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::is_scalar;

using namespace::testing;




class ToolsInputVectors : public Test {
public:
	Tools tools;
	VectorXd vectorXInput;
	vector<VectorXd> vectorVectorXInput1;
	vector<VectorXd> vectorVectorXInput2;
};



TEST_F(ToolsInputVectors, JacobianRaisesExceptionForZeroDenominator) {
	vectorXInput.resize(4);
	vectorXInput << 0, 0, 1, 1;
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), ZeroDivideException);
}

TEST_F(ToolsInputVectors, JacobianExpects4dVectorInput) {
	vectorXInput.resize(3);
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), WrongDimInputException);
}

TEST_F(ToolsInputVectors, JacobianReturns3x4Matrix) {
	vectorXInput.resize(4);
	
	ASSERT_THAT(tools.CalculateJacobian(vectorXInput).rows(), 3);
	ASSERT_THAT(tools.CalculateJacobian(vectorXInput).cols(), 4);
}


//Throws null element exception

TEST_F(ToolsInputVectors, JacobianCompletesNonTrivialExample) {
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

TEST_F(ToolsInputVectors, JacobianThrowsOnNullElementInInput) {
	vectorXInput.resize(4);
	vectorXInput << 1, 2, std::numeric_limits<double>::quiet_NaN(), 0.3;

	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), NullElementVectorException);
}



TEST_F(ToolsInputVectors, JacobianRaisesExceptionOnNull) {
	ASSERT_THROW(tools.CalculateJacobian(vectorXInput), WrongDimInputException);
}

TEST_F(ToolsInputVectors, CalculateRMSERaisesExceptionOnNull) {
	ASSERT_THROW(tools.CalculateRMSE(vectorVectorXInput1, vectorVectorXInput2), WrongDimInputException);
}



int main(int argc, char** argv)
{
	testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
