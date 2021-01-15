#ifndef PERSPECTIVE_MAT_HPP_
#define PERSPECTIVE_MAT_HPP_
#pragma warning(disable:4996)

//#include <vector>
//#include <cmath>

//#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/vector_expression.hpp>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/vector_proxy.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
//#include <boost/numeric/ublas/io.hpp>

using namespace boost;
using namespace boost::numeric::ublas;
using namespace std;

template<typename WorkingType = double>
class PerspectiveTransformation
{
public:
	matrix<WorkingType> Wa;
	std::vector<std::vector<WorkingType>> refPositions;
	matrix<WorkingType> inv_delta_c;

	PerspectiveTransformation()
	{}

	PerspectiveTransformation(std::vector<std::vector<WorkingType> > positions, std::vector<std::vector<WorkingType> > values)
	{
		matrix<WorkingType> L;

		refPositions = positions;

		const int numPoints((int)refPositions.size());
		const int WaLength(8);

		L = matrix<WorkingType>(WaLength, WaLength);

		// Calculate K and store in L
		for (int i(0); i < WaLength/2; i++)
		{
			L(i * 2, 0) = positions[i][0];
			L(i * 2, 1) = positions[i][1];
			L(i * 2, 2) = 1.0;
			L(i * 2, 3) = 0.0;
			L(i * 2, 4) = 0.0;
			L(i * 2, 5) = 0.0;
			L(i * 2, 6) = -positions[i][0] * values[i][0];
			L(i * 2, 7) = -positions[i][1] * values[i][1];

			L(i * 2 + 1, 0) = 0.0;
			L(i * 2 + 1, 1) = 0.0;
			L(i * 2 + 1, 2) = 0.0;
			L(i * 2 + 1, 3) = positions[i][0];
			L(i * 2 + 1, 4) = positions[i][1];
			L(i * 2 + 1, 5) = 1.0;
			L(i * 2 + 1, 6) = -positions[i][0] * values[i][1];
			L(i * 2 + 1, 7) = -positions[i][1] * values[i][0];
		}

		// Solve L^-1 Y = W^T

		typedef permutation_matrix<std::size_t> pmatrix;

		matrix<WorkingType> A(L);
		pmatrix pm(A.size1());
		int res = (int)lu_factorize(A, pm);
		if (res != 0)
		{
			throw;
		}

		matrix<WorkingType> invL(identity_matrix<WorkingType>(A.size1()));
		lu_substitute(A, pm, invL);
		inv_delta_c = invL;

		Wa = matrix<WorkingType>(WaLength, 1);

		matrix <WorkingType> Y(WaLength, 1);
		int i(0);
		for (; i < WaLength / 2; i++)
		{
			Y(i, 0) = values[i][0];
			Y(i+1, 0) = values[i][1];
		}

		Wa = prod(invL, Y);
	}
};

#endif /* PERSPECTIVE_MAT_HPP_ */
