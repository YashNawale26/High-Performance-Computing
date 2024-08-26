#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

#define N 10  // Size of vectors for simplicity

void sort_ascending(std::vector<int>& arr) {
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < arr.size() - 1; i++) {
        for (int j = 0; j < arr.size() - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void sort_descending(std::vector<int>& arr) {
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < arr.size() - 1; i++) {
        for (int j = 0; j < arr.size() - i - 1; j++) {
            if (arr[j] < arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void print_array(const std::vector<int>& arr, const std::string& name) {
    std::cout << name << ": ";
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> a(N), b(N);
    long long min_product = 0;

    // Initialize arrays with simple values for easy calculation
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;  // Simple values from 1 to N
        b[i] = N - i;  // Simple values from N to 1
    }

    // Print the initial arrays
    print_array(a, "Array a");
    print_array(b, "Array b");

    double start_time = omp_get_wtime();

    // Sort array 'a' in ascending order and 'b' in descending order
    #pragma omp sections nowait
    {
        #pragma omp section
        sort_ascending(a);

        #pragma omp section
        sort_descending(b);
    }

    // Print the sorted arrays
    print_array(a, "Sorted array a");
    print_array(b, "Sorted array b");

    // Calculate the minimum scalar product
    #pragma omp parallel for reduction(+:min_product) schedule(guided)
    for (int i = 0; i < N; i++) {
        min_product += static_cast<long long>(a[i]) * b[i];
    }

    double end_time = omp_get_wtime();

    std::cout << "Minimum scalar product: " << min_product << std::endl;
    std::cout << "Time taken: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;
}
