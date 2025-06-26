#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define N 1000 // розмір системи

void generate_matrix(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    srand(0); // фіксований seed для повторюваності
    for (int i = 0; i < N; ++i) {
        b[i] = rand() % 100;
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10;
        }
        A[i][i] += 500; // забезпечення гарної обумовленості
    }
}

void gaussian_elimination(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    for (int k = 0; k < N; ++k) {
        // Нормалізація k-го рядка
        double pivot = A[k][k];
        for (int j = k; j < N; ++j)
            A[k][j] /= pivot;
        b[k] /= pivot;

        // Паралельне обнулення нижче k-го рядка
        #pragma omp parallel for
        for (int i = k + 1; i < N; ++i) {
            double factor = A[i][k];
            for (int j = k; j < N; ++j)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }
}

void back_substitution(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    for (int i = N - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < N; ++j)
            x[i] -= A[i][j] * x[j];
    }
}

int main() {
    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<double> b(N), x(N);

    generate_matrix(A, b);

    double start = omp_get_wtime();
    gaussian_elimination(A, b);
    back_substitution(A, b, x);
    double end = omp_get_wtime();

    std::cout << "Elapsed time: " << end - start << " seconds\n";
    return 0;
}
