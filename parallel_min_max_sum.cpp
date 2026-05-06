// Simple Sequential and Parallel Min, Max, Sum
// Compile: g++ file.cpp -fopenmp

#include<iostream>
#include<omp.h>
#include<climits>

using namespace std;

int main()
{
    int n, arr[100];

    cout << "Enter number of elements: ";
    cin >> n;

    cout << "Enter elements:\n";

    for(int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // ---------- Sequential Min ----------
    int smin = INT_MAX;

    double t1 = omp_get_wtime();

    for(int i = 0; i < n; i++)
    {
        if(arr[i] < smin)
        {
            smin = arr[i];
        }
    }

    double t2 = omp_get_wtime();

    // ---------- Sequential Max ----------
    int smax = INT_MIN;

    double t3 = omp_get_wtime();

    for(int i = 0; i < n; i++)
    {
        if(arr[i] > smax)
        {
            smax = arr[i];
        }
    }

    double t4 = omp_get_wtime();

    // ---------- Sequential Sum ----------
    int ssum = 0;

    double t5 = omp_get_wtime();

    for(int i = 0; i < n; i++)
    {
        ssum += arr[i];
    }

    double t6 = omp_get_wtime();

    // ---------- Parallel Min ----------
    int pmin = INT_MAX;

    double t7 = omp_get_wtime();

    #pragma omp parallel for reduction(min:pmin)
    for(int i = 0; i < n; i++)
    {
        if(arr[i] < pmin)
        {
            pmin = arr[i];
        }
    }

    double t8 = omp_get_wtime();

    // ---------- Parallel Max ----------
    int pmax = INT_MIN;

    double t9 = omp_get_wtime();

    #pragma omp parallel for reduction(max:pmax)
    for(int i = 0; i < n; i++)
    {
        if(arr[i] > pmax)
        {
            pmax = arr[i];
        }
    }

    double t10 = omp_get_wtime();

    // ---------- Parallel Sum ----------
    int psum = 0;

    double t11 = omp_get_wtime();

    #pragma omp parallel for reduction(+:psum)
    for(int i = 0; i < n; i++)
    {
        psum += arr[i];
    }

    double t12 = omp_get_wtime();

    // ---------- Output ----------

    cout << "\nSequential Min = " << smin;
    cout << "\nTime = " << t2 - t1;

    cout << "\n\nSequential Max = " << smax;
    cout << "\nTime = " << t4 - t3;

    cout << "\n\nSequential Sum = " << ssum;
    cout << "\nTime = " << t6 - t5;

    cout << "\n\nParallel Min = " << pmin;
    cout << "\nTime = " << t8 - t7;

    cout << "\n\nParallel Max = " << pmax;
    cout << "\nTime = " << t10 - t9;

    cout << "\n\nParallel Sum = " << psum;
    cout << "\nTime = " << t12 - t11;

    return 0;
}
