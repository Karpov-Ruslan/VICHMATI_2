#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>

constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};
constexpr long double EPSILON = 1e-10;
using matrix = typename boost::numeric::ublas::matrix<long double>;
using vector = typename boost::numeric::ublas::vector<long double>;
using boost::numeric::ublas::prod;
using boost::numeric::ublas::norm_1;

void print(const matrix& A) {
    std::cout << "--------------------\n";
    for (int i = 0; i < A.size1(); i++) {
        for (int j = 0; j < A.size2(); j++) {
            std::cout << A(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------\n";
}

void print(const vector& A) {
    std::cout << "********************\n";
    for (int i = 0; i < A.size(); i++) {
        std::cout << A(i) << "\t";
    }
    std::cout << "\n";
    std::cout << "********************\n";
}

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

bool InvertMatrix (const matrix& input, matrix& inverse) {
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;
    // create a working copy of the input
    matrix A(input);
    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if( res != 0 ) return false;

    // create identity matrix of "inverse"
    inverse.assign(boost::numeric::ublas::identity_matrix(A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(A, pm, inverse);

    return true;
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
    fin.open("data_Jacobi.csv");
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

vector Seidel(const matrix& A, const vector& f) {
    std::ofstream fin;
    vector x(100, 0);

    matrix invLD(100, 100, 0.0L), U = U_matrix(A);
    InvertMatrix(L_matrix(A) + D_matrix(A), invLD);
    unsigned int counter = 0;
    fin.open("data_Seidel.csv");
    fin << std::setprecision(max_precision);
    fin << "Residual;" << norm_1(f - prod(A, x));

    bool expert = true;
    while(expert) {
        vector new_x = prod(invLD ,f - prod(U, x));
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

vector Relax(const matrix& A, const vector& f, const long double w = 1.2L) {
    std::ofstream fin;
    vector x(100, 0);

    matrix inv_wL_plus_D(100, 100, 0.0L), wD_minus_D_plus_wU = (w - 1.0L)*D_matrix(A) + w*U_matrix(A);
    InvertMatrix(w*L_matrix(A) + D_matrix(A), inv_wL_plus_D);
    unsigned int counter = 0;
    fin.open("data_Relax.csv");
    fin << std::setprecision(max_precision);
    fin << "Residual;" << norm_1(f - prod(A, x));

    bool expert = true;
    while(expert) {
        vector new_x = prod(inv_wL_plus_D, w*f - prod(wD_minus_D_plus_wU, x));
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

vector Gauss(const matrix& A, const vector& f) {
    if (A.size1() == A.size2() == f.size() == 1) {
        vector solution(1, f(0)/A(0, 0));
        return solution;
    }

    const auto find_main_element = [](const matrix& A){
        std::tuple main_element = std::make_tuple<long double, int, int>(0.0L, 0, 0);
        for (int i = 0; i < A.size1(); i++) {
            for (int j = 0; j < A.size2(); j++) {
                if (std::abs(std::get<0>(main_element)) < std::abs(A(i, j))) {
                    main_element = std::make_tuple(A(i, j), i, j);
                }
            }
        }
        return main_element;
    };

    std::tuple main_element = find_main_element(A);
    const long double main_element_value = std::get<0>(main_element);
    const int main_pos_i = std::get<1>(main_element);
    const int main_pos_j = std::get<2>(main_element);

    matrix new_A(A.size1() - 1, A.size2() - 1, 0);
    vector new_f(f.size() - 1, 0);
    for (int i = 0; i < A.size1(); i++) {
        if (i == main_pos_i) {continue;}
        const long double coef = -A(i, main_pos_j)/main_element_value;
        int pos_i = i;
        if (i > main_pos_i) {
            pos_i--;
        }

        for (int j = 0; j < A.size2(); j++) {
            if (j == main_pos_j) {continue;}

            int pos_j = j;
            if (j > main_pos_j) {
                pos_j--;
            }

            new_A(pos_i, pos_j) = A(i, j) + coef*A(main_pos_i, j);
        }

        new_f(pos_i) = f(i) + coef*f(main_pos_i);
    }

//    print(new_A);
//    print(new_f);
    vector incomplete_solution = Gauss(new_A, new_f);

    vector complete_solution(incomplete_solution.size() + 1);

    {
        complete_solution(main_pos_j) = f(main_pos_i);
        for (int j = 0; j < A.size2(); j++) {
            if (j == main_pos_j) { continue; }
            int pos_j = j;
            if (j > main_pos_j) {
                pos_j--;
            }
            complete_solution(main_pos_j) -= A(main_pos_i, j) * incomplete_solution(pos_j);
        }
        complete_solution(main_pos_j) /= A(main_pos_i, main_pos_j);

        for (int i = 0; i < complete_solution.size(); i++) {
            if (i == main_pos_j) {continue;}
            int pos_i = i;
            if (i > main_pos_j) {
                pos_i--;
            }
            complete_solution(i) = incomplete_solution(pos_i);
        }
    }

    return complete_solution;
}

vector low_triangular_solution(const matrix& L, const vector& f) {
    vector solution(L.size2(), 0.0L);
    solution(0) = f(0);

    for (int i = 1; i < f.size(); i++) {
        long double sum = 0.0L;
        for (int k = 0; k <= i - 1; k++) {
            sum += L(i, k)*solution(k);
        }
        solution(i) = f(i) - sum;
    }

    return solution;
}

vector up_triangular_solution(const matrix& U, const vector& f) {
    const int max_it = (U.size1() - 1);
    vector solution(U.size1(), 0.0L);
    solution(max_it) = f(max_it)/U(max_it, max_it);
    for (int i = max_it - 1; i >= 0; i--) {
        long double sum = 0.0L;
        for (int k = max_it; k > i; k--) {
            sum += U(i, k)*solution(k);
        }
        solution(i) = (f(i) - sum)/U(i, i);
    }
    return solution;
}

vector LU_decomposition(const matrix& A, const vector& f) {
    const auto LU_decompose = [](const matrix& A){
        matrix L(A.size1(), A.size2(), 0.0L), U(A.size1(), A.size2(), 0.0L);

        for (int i = 0; i < A.size1(); i++) {
            L(i, i) = 1.0L;
            U(0, i) = A(0, i);
        }
        for (int i = 1; i < A.size1(); i++) {
            for (int j = 0; j < A.size2(); j++) {
                if (i <= j) {
                    long double sum = 0.0L;
                    for (int k = 0; k <= i; k++) {
                        sum += L(i, k)*U(k, j);
                    }
                    U(i, j) = A(i, j) - sum;
                }
                else {
                    long double sum = 0.0L;
                    for (int k = 0; k <= j; k++) {
                        sum += L(i, k)*U(k, j);
                    }
                    L(i, j) = (A(i, j) - sum)/U(j, j);
                }
            }
        }
        return std::make_tuple(L, U);
    };

    std::tuple LU = LU_decompose(A);
    const matrix& L = std::get<0>(LU);
    const matrix& U = std::get<1>(LU);

    return up_triangular_solution(U, low_triangular_solution(L, f));
}

int main() {
    // A*x=f
    matrix A(100, 100, 0.0L);
    vector f(100);
    init(A, f);

    Jacobi(A, f);
    Seidel(A, f);
    Relax(A, f, 1.2L);

    std::ofstream fin;
    fin.open("direct_methods.csv");
    fin << "Method;Gauss;LU\nResidual";

    vector sol_Gauss = Gauss(A, f);
    vector sol_LU = LU_decomposition(A, f);
    fin << ";" << norm_1(prod(A, sol_Gauss) - f);
    fin << ";" << norm_1(prod(A, sol_LU) - f) << "\n\n";
    fin << "Answers:;" << sol_Gauss(0) << ";" << sol_LU(0);
    for (int i = 1; i < sol_Gauss.size(); i++) {
        fin << "\n;" << sol_Gauss(i) << ";" << sol_LU(i);
    }

    fin.close();

    return 0;
}
