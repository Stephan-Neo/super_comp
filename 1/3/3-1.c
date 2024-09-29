#include <omp.h>
#include <stdio.h>

#define M 10  // Количество строк матрицы A
#define N 10  // Количество столбцов матрицы A и строк матрицы B
#define K 10  // Количество столбцов матрицы B

int main() {
    float A[M][N], B[N][K], C[M][K];
    int i, j, k;

    // Инициализация матриц A и B
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (i + 1) * (j + 1);
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < K; j++) {
            B[i][j] = (i + 1) + (j + 1);
        }
    }

    // Инициализация матрицы C нулями
    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            C[i][j] = 0.0;
        }
    }

    // Параллельное умножение матриц с выводом информации о том, какой поток обрабатывает какую строку
    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < M; i++) {
        int rank = omp_get_thread_num();
        printf("Поток %d обрабатывает строку %d\n", rank, i);

        for (j = 0; j < K; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("\nРезультирующая матрица C:\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
