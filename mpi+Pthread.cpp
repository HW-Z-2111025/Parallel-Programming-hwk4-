#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pthread.h>
#include <semaphore.h>
using namespace std;

const int N = 1000;
const int worker_count = 2;
float m[N][N];
sem_t sem_leader;
sem_t sem_Divsion[worker_count - 1];
sem_t sem_Elimination[worker_count - 1];

struct threadParam_t {
    int t_id;
};

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

int r1, r2;
int comm_sz;
int my_rank;

void* threadFunc(void* param) {
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
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
        } else {
            sem_wait(&sem_Divsion[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < worker_count - 1; i++)
                sem_post(&sem_Divsion[i]);
        }

        int i;
        if ((r1 <= k + 1) && (k + 1 <= r2))
            i = k + 1;
        if (k + 1 < r1) i = r1;
        if (k + 1 > r2) i = N;

        for (i = i + t_id; i < r2; i += worker_count) {
            for (int j = k + 1; j < N; ++j)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }

        if (t_id == 0) {
            for (int i = 0; i < worker_count - 1; i++)
                sem_wait(&sem_leader);
            for (int i = 0; i < worker_count - 1; i++)
                sem_post(&sem_Elimination[i]);
        } else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
    return NULL;
}

int main() {
    LARGE_INTEGER start, end, frequency;
    double timecount;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    r1 = my_rank * (N - N % comm_sz) / comm_sz;
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

    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < worker_count - 1; ++i) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    pthread_t* handles = new pthread_t[worker_count];
    threadParam_t* param = new threadParam_t[worker_count];
    for (int t_id = 0; t_id < worker_count; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (&param[t_id]));
    }

    for (int t_id = 0; t_id < worker_count; t_id++)
        pthread_join(handles[t_id], NULL);

    sem_destroy(&sem_leader);
    for (int t_id = 0; t_id < worker_count; t_id++) {
        sem_destroy(&sem_Divsion[t_id - 1]);
        sem_destroy(&sem_Elimination[t_id - 1]);
    }
    delete[] handles;
    delete[] param;

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

    MPI_Finalize();
    QueryPerformanceCounter(&end);
    timecount = (end.QuadPart - start.QuadPart) * 1000000.0 / frequency.QuadPart;
    cout << timecount << " μs" << endl;
    return 0;
}
