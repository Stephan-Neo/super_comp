#include <omp.h>
#include <stdio.h>

#define M 10  // Размер матрицы и вектора

int main() {
    float A[M][M], b[M], c[M];
    int i, j, rank;

    // Инициализация данных
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = (j + 1) * 1.0;
        }
        b[i] = 1.0 * (i + 1);
        c[i] = 0.0;
    }

    printf("Матрица A и вектор b:\n");
    for (i = 0; i < M; i++) {
        printf("A[%d] = ", i);
        for (j = 0; j < M; j++) {
            printf("%.1f ", A[i][j]);
        }
        printf(" b[%d] = %.1f\n", i, b[i]);
    }

    // Параллельное умножение матрицы на вектор с использованием секций
    #pragma omp parallel shared(A, b, c) private(i, j, rank)
    {
        rank = omp_get_thread_num();
        #pragma omp sections
        {
            // Секция 1: Обработка первой половины строк
            #pragma omp section
            {
                for (i = 0; i < M / 2; i++) {
                    for (j = 0; j < M; j++) {
                        c[i] += A[i][j] * b[j];
                    }
                    printf("Секция 1, поток %d: c[%d] = %.2f\n", rank, i, c[i]);
                }
            }

            // Секция 2: Обработка второй половины строк
            #pragma omp section
            {
                for (i = M / 2; i < M; i++) {
                    for (j = 0; j < M; j++) {
                        c[i] += A[i][j] * b[j];
                    }
                    printf("Секция 2, поток %d: c[%d] = %.2f\n", rank, i, c[i]);
                }
            }
        }
    }

    // Вывод результата
    printf("\nРезультирующий вектор c:\n");
    for (i = 0; i < M; i++) {
        printf("c[%d] = %.2f\n", i, c[i]);
    }

    return 0;
}
