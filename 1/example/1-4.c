#include <omp.h>
#include<stdio.h>
#define N     50

int main () {
    int i, size, rank;
    float a[N], b[N], c[N];
    /* Инициализация элементов векторов */
    for (i=0; i < N; i++)
    a[i] = b[i] = i * 1.0;
    /* Создание множества параллельных процессов */
    #pragma omp parallel shared(a,b,c) private(i,rank,size)
    {
    /* Каждый процесс находит свой порядковый номер и выводит его на экран */
    rank = omp_get_thread_num();
    printf("Thread %d starting...\n",rank);
    /* Директива задания секций */
    #pragma omp sections nowait
        {
    /* Секция 0*/ 
        #pragma omp section
        for (i=0; i < N/2; i++)
            {
            c[i] = a[i] + b[i];
            printf("rank = %d i= %d c[i]= %f\n", rank,i,c[i]);
            }
    /* Секция 1*/ 
        #pragma omp section
        for (i=N/2; i < N; i++)
            {
            c[i] = a[i] + b[i];
            printf("rank = %d i= %d c[i]= %f\n", rank,i,c[i]);
            }
        }  /* Завершение блока секций */
    if (rank == 0)
        {
        size = omp_get_num_threads();
        printf("Number of threads = %d\n", size);
        }
    }  /* Завершение параллельной части */
}