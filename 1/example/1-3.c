#include <omp.h>
#include<stdio.h>
#define VECLEN 100
float a[VECLEN], b[VECLEN], sum;

float dotprod () {
    int i,rank;
    rank = omp_get_thread_num();

    #pragma omp for reduction(+:sum)
    for (i=0; i < VECLEN; i++)
        {
        sum = sum + (a[i]*b[i]);
        printf(" rank = %d i=%d\n", rank, i);
        }

    return(sum);
}

int main () {
    int i;
    for (i=0; i < VECLEN; i++)
    a[i] = b[i] = 1.0 * i;
    sum = 0.0;
    
    #pragma omp parallel
    sum = dotprod();
    printf("Sum = %f\n",sum);
}