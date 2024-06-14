#include <windows.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

const int N = 1000;
float matrix[N][N];

void m_reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++)
            matrix[i][j] = 0;
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            matrix[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                matrix[i][j] += matrix[k][j];
}

int main() {
    LARGE_INTEGER frequency;        // 频率
    LARGE_INTEGER start, end;       // 开始和结束时间
    double timecount = 0;

    // 获取计时器频率
    QueryPerformanceFrequency(&frequency);

    m_reset();

    // 开始计时
    QueryPerformanceCounter(&start);

    for (int cy = 0; cy < 20; cy++) {
        for (int k = 0; k < N; k++) {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            matrix[k][k] = 1.0;
            for (int i = k + 1; i < N; i++) {
                for (int j = k + 1; j < N; j++)
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                matrix[i][k] = 0;
            }
        }
    }

    // 结束计时
    QueryPerformanceCounter(&end);

    // 计算时间差（以微秒为单位）
    timecount = (double)(end.QuadPart - start.QuadPart) * 1000000.0 / frequency.QuadPart;

    // 输出平均时间
    cout << timecount / 20 << " microseconds" << endl;

    return 0;
}