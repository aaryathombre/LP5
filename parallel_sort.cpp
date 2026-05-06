// Simple Parallel Bubble Sort and Merge Sort
// Compile: g++ file.cpp -fopenmp

#include<iostream>
#include<omp.h>
using namespace std;

// ---------- Parallel Bubble Sort ----------
void bubbleSort(int arr[], int n)
{
    for(int i = 0; i < n - 1; i++)
    {
        #pragma omp parallel for
        for(int j = 0; j < n - i - 1; j++)
        {
            if(arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// ---------- Merge Function ----------
void merge(int arr[], int low, int mid, int high)
{
    int temp[100];

    int i = low;
    int j = mid + 1;
    int k = low;

    while(i <= mid && j <= high)
    {
        if(arr[i] < arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    while(i <= mid)
    {
        temp[k++] = arr[i++];
    }

    while(j <= high)
    {
        temp[k++] = arr[j++];
    }

    for(i = low; i <= high; i++)
    {
        arr[i] = temp[i];
    }
}

// ---------- Parallel Merge Sort ----------
void mergeSort(int arr[], int low, int high)
{
    if(low < high)
    {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, low, mid);

            #pragma omp section
            mergeSort(arr, mid + 1, high);
        }

        merge(arr, low, mid, high);
    }
}

// ---------- Display ----------
void display(int arr[], int n)
{
    for(int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
}

// ---------- Main ----------
int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int arr1[100], arr2[100];

    cout << "Enter elements:\n";

    for(int i = 0; i < n; i++)
    {
        cin >> arr1[i];

        arr2[i] = arr1[i];
    }

    // Bubble Sort
    bubbleSort(arr1, n);

    cout << "\nParallel Bubble Sort:\n";
    display(arr1, n);

    // Merge Sort
    mergeSort(arr2, 0, n - 1);

    cout << "\n\nParallel Merge Sort:\n";
    display(arr2, n);

    return 0;
}
