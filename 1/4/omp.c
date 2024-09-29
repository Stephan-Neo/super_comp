#include <omp.h>
#include <stdio.h>

#define M 300  // Размер матрицы и вектора

int main() {
    float A[M][M], b[M], c[M];
    int i, j;

    // Инициализация данных
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = (i + 1) * (j + 1);
        }
        b[i] = 1.0;
        c[i] = 0.0;
    }

    // Замер времени выполнения
    double start_time = omp_get_wtime();

    #pragma omp parallel for private(i, j)
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            c[i] += A[i][j] * b[j];
        }
    }

    double end_time = omp_get_wtime();
    printf("Время выполнения (OpenMP): %f секунд\n", end_time - start_time);

    printf("Результат умножения матрицы на вектор (OpenMP):\n");
    for (i = 0; i < M; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");

    return 0;
}
