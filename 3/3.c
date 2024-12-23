#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int ProcNum = 0;      // Number of available processes
int ProcRank = 0;

void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < Size; i++) {
        pVector[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < Size; j++) {
            pMatrix[i * Size + j] = (double)rand() / RAND_MAX;
        }
    }
}

void ResultReplication(double* pProcResult, double* pResult, int Size, int RowNum) {
    int* pReceiveNum = (int*)malloc(ProcNum * sizeof(int));
    int* pReceiveInd = (int*)malloc(ProcNum * sizeof(int));
    int RestRows = Size;

    pReceiveInd[0] = 0;
    pReceiveNum[0] = Size / ProcNum;
    for (int i = 1; i < ProcNum; i++) {
        RestRows -= pReceiveNum[i - 1];
        pReceiveNum[i] = RestRows / (ProcNum - i);
        pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
    }

    MPI_Allgatherv(pProcResult, RowNum, MPI_DOUBLE, pResult, pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

    free(pReceiveNum);
    free(pReceiveInd);
}

void ParallelResultCalculation(double* pProcRows, double* pVector, double* pProcResult, int Size, int RowNum) {
    for (int i = 0; i < RowNum; i++) {
        pProcResult[i] = 0.0;
        for (int j = 0; j < Size; j++) {
            pProcResult[i] += pProcRows[i * Size + j] * pVector[j];
        }
    }
}

void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, int Size, int RowNum) {
    int* pSendNum = (int*)malloc(ProcNum * sizeof(int));
    int* pSendInd = (int*)malloc(ProcNum * sizeof(int));
    int RestRows = Size;

    MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    RowNum = Size / ProcNum;
    pSendNum[0] = RowNum * Size;
    pSendInd[0] = 0;
    for (int i = 1; i < ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows / (ProcNum - i);
        pSendNum[i] = RowNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }

    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(pSendNum);
    free(pSendInd);
}

void ProcessInitialization(double** pMatrix, double** pVector, double** pResult, double** pProcRows, double** pProcResult, int* Size, int* RowNum) {
    int RestRows;

    if (ProcRank == 0) {
        do {
            printf("Input matrix size: ");
            scanf("%d", Size);
            if (*Size < ProcNum) {
                printf("Matrix size must be greater than the number of processes!\n");
            }
        } while (*Size < ProcNum);
    }
    MPI_Bcast(Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    RestRows = *Size;
    for (int i = 0; i < ProcRank; i++) {
        RestRows -= RestRows / (ProcNum - i);
    }
    *RowNum = RestRows / (ProcNum - ProcRank);

    *pVector = (double*)malloc(*Size * sizeof(double));
    *pResult = (double*)malloc(*Size * sizeof(double));
    *pProcRows = (double*)malloc((*RowNum) * (*Size) * sizeof(double));
    *pProcResult = (double*)malloc(*RowNum * sizeof(double));

    if (ProcRank == 0) {
        *pMatrix = (double*)malloc((*Size) * (*Size) * sizeof(double));
        RandomDataInitialization(*pMatrix, *pVector, *Size);
    }
}

void ProcessTermination(double* pMatrix, double* pVector, double* pResult, double* pProcRows, double* pProcResult) {
    if (ProcRank == 0) {
        free(pMatrix);
    }
    free(pVector);
    free(pResult);
    free(pProcRows);
    free(pProcResult);
}

int main(int argc, char* argv[]) {
    double* pMatrix;
    double* pVector;
    double* pResult;
    int Size;
    double* pProcRows;
    double* pProcResult;
    int RowNum;
    double Start, Finish, Duration;
    double CalcStart, CalcFinish, CalcDuration;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    Start = MPI_Wtime();

    ProcessInitialization(&pMatrix, &pVector, &pResult, &pProcRows, &pProcResult, &Size, &RowNum);
    DataDistribution(pMatrix, pProcRows, pVector, Size, RowNum);

    CalcStart = MPI_Wtime();
    ParallelResultCalculation(pProcRows, pVector, pProcResult, Size, RowNum);
    CalcFinish = MPI_Wtime();
    CalcDuration = CalcFinish - CalcStart;

    ResultReplication(pProcResult, pResult, Size, RowNum);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    if (ProcRank == 0) {
        printf("Program execution time: %.6f seconds\n", Duration);
        printf("Matrix-vector multiplication time: %.6f seconds\n", CalcDuration);
    }

    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcResult);

    MPI_Finalize();
    return 0;
}
