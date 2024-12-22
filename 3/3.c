#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Прототипы функций
void ProcessInitialization(double** pMatrix, double** pVector, double** pResult, 
                           double** pProcRows, double** pProcResult, int* Size, int* RowNum, 
                           int ProcNum, int ProcRank);
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, 
                      int Size, int RowNum, int ProcNum, int ProcRank);
void ParallelResultCalculation(double* pProcRows, double* pVector, double* pProcResult, 
                                int Size, int RowNum);
void ResultReplication(double* pProcResult, double* pResult, int Size, int RowNum, 
                       int ProcNum, int ProcRank);
void ProcessTermination(double* pMatrix, double* pVector, double* pResult, 
                        double* pProcRows, double* pProcResult, int ProcRank);

int main(int argc, char* argv[]) {
    double *pMatrix = NULL;  // Исходная матрица
    double *pVector = NULL;  // Исходный вектор
    double *pResult = NULL;  // Результат умножения матрицы на вектор 
    double *pProcRows = NULL;
    double *pProcResult = NULL;
    int Size;                // Размер матрицы и вектора
    int RowNum;              // Количество строк, обрабатываемых процессом
    int ProcNum, ProcRank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    // Инициализация
    ProcessInitialization(&pMatrix, &pVector, &pResult, &pProcRows, &pProcResult, &Size, &RowNum, ProcNum, ProcRank);

    // Распределение данных
    DataDistribution(pMatrix, pProcRows, pVector, Size, RowNum, ProcNum, ProcRank);

    // Параллельное вычисление
    ParallelResultCalculation(pProcRows, pVector, pProcResult, Size, RowNum);

    // Сбор результатов
    ResultReplication(pProcResult, pResult, Size, RowNum, ProcNum, ProcRank);

    // Завершение
    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcResult, ProcRank);

    MPI_Finalize();
    return 0;
}

void ProcessInitialization(double** pMatrix, double** pVector, double** pResult, 
                           double** pProcRows, double** pProcResult, int* Size, int* RowNum, 
                           int ProcNum, int ProcRank) {
    if (ProcRank == 0) {
        do {
            printf("\nВведите размер матрицы: ");
            scanf("%d", Size);
            if (*Size < ProcNum) {
                printf("Размер матрицы должен превышать количество процессов!\n");
            }
        } while (*Size < ProcNum);
    }

    MPI_Bcast(Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int RestRows = *Size;
    for (int i = 0; i < ProcRank; i++) {
        RestRows -= RestRows / (ProcNum - i);
    }
    *RowNum = RestRows / (ProcNum - ProcRank);

    *pVector = (double*)malloc(*Size * sizeof(double));
    *pResult = (double*)malloc(*Size * sizeof(double));
    *pProcRows = (double*)malloc(*RowNum * *Size * sizeof(double));
    *pProcResult = (double*)malloc(*RowNum * sizeof(double));

    if (ProcRank == 0) {
        *pMatrix = (double*)malloc(*Size * *Size * sizeof(double));
        for (int i = 0; i < (*Size) * (*Size); i++) {
            (*pMatrix)[i] = rand() % 100;  // Заполнение матрицы случайными числами
        }
        for (int i = 0; i < *Size; i++) {
            (*pVector)[i] = rand() % 100;  // Заполнение вектора случайными числами
        }
    }
}

void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, 
                      int Size, int RowNum, int ProcNum, int ProcRank) {
    int *pSendNum = (int*)malloc(ProcNum * sizeof(int));
    int *pSendInd = (int*)malloc(ProcNum * sizeof(int));

    MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int RestRows = Size;
    for (int i = 0; i < ProcNum; i++) {
        int Rows = RestRows / (ProcNum - i);
        pSendNum[i] = Rows * Size;
        pSendInd[i] = (i == 0) ? 0 : pSendInd[i - 1] + pSendNum[i - 1];
        RestRows -= Rows;
    }

    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, 
                 pProcRows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(pSendNum);
    free(pSendInd);
}

void ParallelResultCalculation(double* pProcRows, double* pVector, double* pProcResult, 
                                int Size, int RowNum) {
    for (int i = 0; i < RowNum; i++) {
        pProcResult[i] = 0.0;
        for (int j = 0; j < Size; j++) {
            pProcResult[i] += pProcRows[i * Size + j] * pVector[j];
        }
    }
}

void ResultReplication(double* pProcResult, double* pResult, int Size, int RowNum, 
                       int ProcNum, int ProcRank) {
    int *pReceiveNum = (int*)malloc(ProcNum * sizeof(int));
    int *pReceiveInd = (int*)malloc(ProcNum * sizeof(int));

    int RestRows = Size;
    for (int i = 0; i < ProcNum; i++) {
        int Rows = RestRows / (ProcNum - i);
        pReceiveNum[i] = Rows;
        pReceiveInd[i] = (i == 0) ? 0 : pReceiveInd[i - 1] + pReceiveNum[i - 1];
        RestRows -= Rows;
    }

    MPI_Allgatherv(pProcResult, RowNum, MPI_DOUBLE, pResult, 
                   pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

    free(pReceiveNum);
    free(pReceiveInd);
}

void ProcessTermination(double* pMatrix, double* pVector, double* pResult, 
                        double* pProcRows, double* pProcResult, int ProcRank) {
    if (ProcRank == 0) {
        free(pMatrix);
        free(pResult);
    }
    free(pVector);
    free(pProcRows);
    free(pProcResult);
}
