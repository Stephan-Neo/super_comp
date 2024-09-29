#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 300  // Размер матрицы и вектора

void matrix_vector_mult(float* A, float* b, float* c, int rows_per_proc, int N) {
    for (int i = 0; i < rows_per_proc; i++) {
        c[i] = 0;
        for (int j = 0; j < N; j++) {
            c[i] += A[i * N + j] * b[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    int rows_per_proc;
    float *A, *b, *local_A, *local_c, *c;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_proc = M / size;

    b = (float*)malloc(M * sizeof(float));
    local_A = (float*)malloc(rows_per_proc * M * sizeof(float));
    local_c = (float*)malloc(rows_per_proc * sizeof(float));
    if (rank == 0) {
        A = (float*)malloc(M * M * sizeof(float));
        c = (float*)malloc(M * sizeof(float));

        // Инициализация матрицы A и вектора b
        for (int i = 0; i < M; i++) {
            b[i] = 1.0;
            for (int j = 0; j < M; j++) {
                A[i * M + j] = (i + 1) * (j + 1);
            }
        }
    }

    // Замер времени начала
    start_time = MPI_Wtime();

    // Передача вектора b всем процессам
    MPI_Bcast(b, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Разделение матрицы A между процессами
    MPI_Scatter(A, rows_per_proc * M, MPI_FLOAT, local_A, rows_per_proc * M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Умножение локальной части матрицы на вектор
    matrix_vector_mult(local_A, b, local_c, rows_per_proc, M);

    // Сбор результатов
    MPI_Gather(local_c, rows_per_proc, MPI_FLOAT, c, rows_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Замер времени завершения
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Результат умножения матрицы на вектор (MPI):\n");
        for (int i = 0; i < M; i++) {
            printf("%.2f ", c[i]);
        }
        printf("\n");
        printf("Время выполнения: %f секунд\n", end_time - start_time);
        free(A);
        free(c);
    }

    free(b);
    free(local_A);
    free(local_c);

    MPI_Finalize();
    return 0;
}
