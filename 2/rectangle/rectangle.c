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

    if (myrank == 0) {
        printf("Enter the number of intervals (n): ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / n;
    for (i = myrank + 1; i <= n; i += nprocs) {
        x = h * (i - 0.5);
        local_sum += f(x);
    }

    local_sum *= 4.0 * h;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        printf("Approximated value of pi: %.16f\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}
