#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

constexpr unsigned int size = 100;
constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};
constexpr long double EPSILON = 1e-10;
using matrix = typename boost::numeric::ublas::matrix<long double>;
using vector = typename boost::numeric::ublas::vector<long double>;
using boost::numeric::ublas::prod;
using boost::numeric::ublas::norm_1;



void init(matrix& A, vector& f) {
    // A-zone
    for (int j = 0; j < 100; j++) {
        A(0, j) = 1.0L;
    }
    for (int i = 1; i < 99; i++) {
        A(i, i) = 10.0L;
        A(i, i - 1) = A(i, i + 1) = 1.0L;
    }
    A(99, 98) = A(99, 99) = 1.0L;
    // f-zone
    for (int i = 0; i < 100; i++) {
        f(i) = static_cast<long double>(100 - i);
    }
}

matrix inverse_diag_matrix(const matrix &D) {
    matrix output(D.size1(), D.size2(), 0.0L);
    for (int i = 0; i < 100; i++) {
        output(i, i) = 1.0L/(D(i, i));
    }
    return output;
}

// A = L + D + U
matrix L_matrix(const matrix& A) {
    matrix L(A.size1(), A.size2(), 0.0L);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            if (j < i) {
                L(i, j) = A(i, j);
            }
        }
    }
    return L;
}

// A = L + D + U
matrix U_matrix(const matrix& A) {
    matrix U(A.size1(), A.size2(), 0.0L);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            if (j > i) {
                U(i, j) = A(i, j);
            }
        }
    }
    return U;
}

// A = L + D + U
matrix D_matrix(const matrix& A) {
    matrix D(A.size1(), A.size2(), 0.0L);
    for (int i = 0; i < 100; i++) {
        D(i, i) = A(i, i);
    }
    return D;
}

vector Jacobi(const matrix& A, const vector& f) {
    std::ofstream fin;
    vector x(100, 0);

    matrix L = L_matrix(A), invD = inverse_diag_matrix(D_matrix(A)), U = U_matrix(A);
    unsigned int counter = 0;
    fin.open("data.csv");
    fin << std::setprecision(max_precision);
    fin << "Residual;" << norm_1(f - prod(A, x));

    bool expert = true;
    while(expert) {
        vector new_x = prod(invD ,f - prod(L + U, x));
        if (norm_1(new_x - x) < EPSILON) {
            expert = false;
        }
        x = new_x;
        counter++;
        fin << ";" << norm_1(f - prod(A, x));
    }

    fin << "\nnumber of iterations";
    for (int i = 0; i <= counter; i++) {
        fin << ";" << i;
    }

    fin << "\n\nAnswer:";
    for (int i = 0; i < 100; i++) {
        fin << "\n" << x(i);
    }

    fin.close();
    return x;
}

int main() {

    // A*x=f
    matrix A(100, 100, 0.0L);
    vector f(100);
    init(A, f);
    std::cout << Jacobi(A, f);

    return 0;
}
