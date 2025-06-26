#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <chrono> // Для вимірювання часу
#include <omp.h>  // Для OpenMP


// Функція для генерації випадкової матриці з домінуючою діагоналлю
void generateRandomMatrix(std::vector<std::vector<double>>& A, std::vector<double>& b, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0); // Для недіагональних елементів

    for (int i = 0; i < n; ++i) {
        double diag_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                // Діагональний елемент буде встановлений пізніше, щоб домінувати
                A[i][j] = 0.0; // Тимчасово
            } else {
                A[i][j] = dis(gen);
                diag_sum += std::abs(A[i][j]);
            }
        }
        // Забезпечуємо діагональну домінантність
        A[i][i] = diag_sum + std::abs(dis(gen)) + 1.0; // Додаємо домінуючий елемент
        if (dis(gen) < 0) A[i][i] *= -1; // Випадковий знак
        b[i] = dis(gen) * n; // Випадкові значення для вектора b
    }
}

// Функція для друку матриці (для налагодження, не для великих матриць)
void printMatrix(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << A[i][j] << " ";
        }
        std::cout << "| " << std::setw(10) << std::fixed << std::setprecision(4) << b[i] << std::endl;
    }
}

// Функція для розв'язання системи методом Гаусса
std::vector<double> gaussianElimination(std::vector<std::vector<double>> A, std::vector<double> b, int n) {
    // Прямий хід (елімінація)
    for (int k = 0; k < n; ++k) {
        // Знаходимо опорний елемент (максимальний за модулем в поточному стовпці)
        int max_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > std::abs(A[max_row][k])) {
                max_row = i;
            }
        }

        // Міняємо рядки, якщо необхідно
        std::swap(A[k], A[max_row]);
        std::swap(b[k], b[max_row]);

        // Нормалізація поточного опорного рядка та елімінація нижче
        #pragma omp parallel for // Паралелізація зовнішнього циклу
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Зворотний хід (підстановка)
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Використання: " << argv[0] << " <розмір_матриці>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]); // Розмір матриці
    if (n <= 0) {
        std::cerr << "Розмір матриці повинен бути позитивним числом." << std::endl;
        return 1;
    }

    // Розрахунок приблизного об'єму пам'яті: (N*N + N) * sizeof(double)
    double memory_in_gb = (static_cast<double>(n) * n + n) * sizeof(double) / (1024 * 1024 * 1024);
    std::cout << "Приблизне використання пам'яті для матриці " << n << "x" << n << ": " << memory_in_gb << " ГБ" << std::endl;

    if (memory_in_gb > 1.0) {
        std::cout << "Увага: Розмір матриці може перевищувати 1 ГБ пам'яті. Розгляньте зменшення N." << std::endl;
        // Можливо, тут варто вийти або запропонувати зменшити N
    }
    
    // Ініціалізація матриць
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<double> b(n);
    
    // Генеруємо однакові матриці для всіх запусків
    // Використовуємо фіксоване зерно для генератора випадкових чисел
    // Це забезпечить однакові матриці для різних запусків з різною кількістю потоків
    // Але краще передавати A та b з попередньо згенерованих файлів для кластера
    // Для демонстрації тут просто згенеруємо один раз
    std::cout << "Генерація випадкової матриці..." << std::endl;
    std::vector<std::vector<double>> A_copy(n, std::vector<double>(n));
    std::vector<double> b_copy(n);
    generateRandomMatrix(A_copy, b_copy, n);
    
    // Зберігаємо копії для перевірки
    A = A_copy;
    b = b_copy;

    std::cout << "Матриця згенерована. Починаємо обчислення..." << std::endl;

    std::vector<double> execution_times;
    
    // Цикл для запуску з різною кількістю процесорів
    for (int num_threads = 1; num_threads <= 8; ++num_threads) {
        omp_set_num_threads(num_threads); // Встановлюємо кількість потоків

        std::vector<std::vector<double>> current_A = A_copy;
        std::vector<double> current_b = b_copy;

        // Вимірювання часу за допомогою omp_get_wtime()
        double start_time = omp_get_wtime(); 
        std::vector<double> x = gaussianElimination(current_A, current_b, n);
        double end_time = omp_get_wtime();
        double duration = end_time - start_time;

        execution_times.push_back(duration); // Зберігаємо час виконання

        std::cout << "\n--- Результати для " << num_threads << " потоків ---" << std::endl;
        std::cout << "Час виконання: " << std::fixed << std::setprecision(6) << duration << " секунд" << std::endl;

        // Перевірка коректності (для невеликих N)
        // Для великих N краще перевіряти норму нев'язки (||Ax - b||)
        if (n <= 10) { // Друк розв'язку тільки для малих матриць
            std::cout << "Розв'язок x:" << std::endl;
            for (int i = 0; i < n; ++i) {
                std::cout << "x[" << i << "] = " << std::fixed << std::setprecision(6) << x[i] << std::endl;
            }
        } else {
             // Обчислення норми нев'язки для перевірки
            double max_error = 0.0;
            for (int i = 0; i < n; ++i) {
                double res = 0.0;
                for (int j = 0; j < n; ++j) {
                    res += A_copy[i][j] * x[j]; // Використовуємо початкову A_copy
                }
                max_error = std::max(max_error, std::abs(res - b_copy[i]));
            }
            std::cout << "\n--- Зведення часів виконання ---" << std::endl;
            for (size_t i = 0; i < execution_times.size(); ++i) {
                std::cout << "Потоків " << (i + 1) << ": " << std::fixed << std::setprecision(6) << execution_times[i] << " сек" << std::endl;
            }
            std::cout << "Максимальна абсолютна помилка (норма нев'язки): " << max_error << std::endl;
            if (max_error < 1e-6) { // Припустима похибка
                 std::cout << "Результати коректні (в межах допустимої похибки)." << std::endl;
            } else {
                std::cout << "Увага: Помилка може бути завеликою. Розгляньте збільшення точності або перевірку алгоритму." << std::endl;
            }
        }
    }

    return 0;
}
