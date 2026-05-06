// Parallel BFS and DFS using Arrays (OpenMP)
// Compile: g++ file.cpp -fopenmp

#include <iostream>
#include <omp.h>
using namespace std;

int graph[10][10], visited[10], n;

// ---------- Parallel BFS ----------
void parallelBFS(int start)
{
    int queue[10], front = 0, rear = 0;

    queue[rear++] = start;
    visited[start] = 1;

    cout << "Parallel BFS: ";

    while (front < rear)
    {
        int current = queue[front++];
        cout << current << " ";

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            if (graph[current][i] == 1 && visited[i] == 0)
            {
                visited[i] = 1;

                #pragma omp critical
                {
                    queue[rear++] = i;
                }
            }
        }
    }
}

// ---------- Parallel DFS ----------
void parallelDFS(int node)
{
    visited[node] = 1;
    cout << node << " ";

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        if (graph[node][i] == 1 && visited[i] == 0)
        {
            parallelDFS(i);
        }
    }
}

// ---------- Main ----------
int main()
{
    int edges, u, v, start;

    cout << "Enter number of vertices: ";
    cin >> n;

    // Initialize graph with 0
    for (int i = 0; i < n; i++)
    {
        visited[i] = 0;

        for (int j = 0; j < n; j++)
        {
            graph[i][j] = 0;
        }
    }

    cout << "Enter number of edges: ";
    cin >> edges;

    cout << "Enter edges (u v):\n";

    for (int i = 0; i < edges; i++)
    {
        cin >> u >> v;

        graph[u][v] = 1;
        graph[v][u] = 1; // Undirected graph
    }

    cout << "Enter starting vertex: ";
    cin >> start;

    // BFS
    parallelBFS(start);

    // Reset visited array
    for (int i = 0; i < n; i++)
    {
        visited[i] = 0;
    }

    cout << "\nParallel DFS: ";

    // DFS
    parallelDFS(start);

    return 0;
}
