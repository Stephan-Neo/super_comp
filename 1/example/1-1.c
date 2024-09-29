#include<omp.h>
#include<stdio.h>

int main() {
    int size, rank;

    #pragma omp parallel private(size, rank)
    {
        rank = omp_get_thread_num();
        printf("Hello World from thread = %d\n", rank);

        if (rank == 0) {
            size = omp_get_num_threads();
            printf("Number of threads = %d\n", size);
        }
    }
    return 0;
}
