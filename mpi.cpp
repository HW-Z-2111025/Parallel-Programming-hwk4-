#include <windows.h>
#include <iostream>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const int N = 600;
float m[N][N];

void m_reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}

int main() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start, end;
    double timecount;
    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int r1 = my_rank * (N - N % comm_sz) / comm_sz;
    int r2;
    if (my_rank != comm_sz - 1)
        r2 = my_rank * (N - N % comm_sz) / comm_sz + (N - N % comm_sz) / comm_sz - 1;
    else
        r2 = N - 1;

    if (my_rank == 0) {
        m_reset();
        for (int i = 1; i < comm_sz; i++)
            MPI_Send(m, N * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(m, N * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (int k = 0; k < N; k++) {
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < N; j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
            for (int num = my_rank + 1; num < comm_sz; num++)
                MPI_Send(&m[k][0], N, MPI_FLOAT, num, 1, MPI_COMM_WORLD);
        } else {
            if (k <= r2)
                MPI_Recv(&m[k][0], N, MPI_FLOAT, k / ((N - N % comm_sz) / comm_sz), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int i;
        if ((r1 <= k + 1) && (k + 1 <= r2))
            i = k + 1;
        if (k + 1 < r1)
            i = r1;
        if (k + 1 > r2)
            i = N;
        for (i; i <= r2; i++) {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }

    if (my_rank != 0)
        MPI_Send(&m[r1][0], N * (r2 - r1 + 1), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    else {
        for (int q = 1; q < comm_sz; q++) {
            if (q != comm_sz - 1)
                MPI_Recv(&m[q * (N - N % comm_sz) / comm_sz][0], N * (r2 - r1 + 1), MPI_FLOAT, q, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else
                MPI_Recv(&m[q * (N - N % comm_sz) / comm_sz][0], N * (r2 - r1 + 1 + N % comm_sz), MPI_FLOAT, q, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    QueryPerformanceCounter(&end);

    timecount = (double)(end.QuadPart - start.QuadPart) * 1000000.0 / frequency.QuadPart;
    cout << timecount << " microseconds" << endl;

    MPI_Finalize();
    return 0;
}
