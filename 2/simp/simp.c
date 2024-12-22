#include <stdio.h>
#include <mpi.h>

double f(double x) {
    return 1.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {
    int n, myrank, nprocs, i;
    double h, local_sum = 0.0, total_sum = 0.0, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Процесс с рангом 0 запрашивает количество интервалов
    if (myrank == 0) {
        printf("Enter the number of intervals (n, even only): ");
        scanf("%d", &n);
        if (n % 2 != 0) {
            printf("Error: n must be an even number.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Передача значения n всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / n;  // Ширина интервала

    // Вычисление частичных сумм на каждом процессе
    for (i = myrank; i <= n; i += nprocs) {
        x = i * h;
        if (i == 0 || i == n) {
            local_sum += f(x);  // Границы добавляются без умножения
        } else if (i % 2 == 0) {
            local_sum += 2 * f(x);  // Узлы с чётным индексом умножаются на 2
        } else {
            local_sum += 4 * f(x);  // Узлы с нечётным индексом умножаются на 4
        }
    }

    local_sum *= h / 3.0;  // Умножение на h/3 в формуле Симпсона

    // Суммирование всех локальных результатов в процесс 0
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Процесс 0 выводит результат
    if (myrank == 0) {
        printf("Approximated value of pi: %.16f\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}
