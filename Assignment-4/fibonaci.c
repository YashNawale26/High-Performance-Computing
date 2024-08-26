/*
#include <omp.h>
#include <stdio.h>


//Parallel Iterative Fibonacci (Using parallel for)
int main() {
    int n = 20;  // Change n to compute up to nth Fibonacci number
    int fib[n + 1];

    // Base cases
    fib[0] = 0;
    fib[1] = 1;

    #pragma omp parallel for
    for (int i = 2; i <= n; i++) {
        fib[i] = fib[i-1] + fib[i-2];
    }

    printf("Fibonacci number F(%d) = %d\n", n, fib[n]);
    return 0;
}
*/

#include <omp.h>
#include <stdio.h>

int fib(int n) {
    int x, y;
    if (n < 2) {
        return n;
    } else {
        #pragma omp task shared(x)
        x = fib(n-1);
        #pragma omp task shared(y)
        y = fib(n-2);
        #pragma omp taskwait
        return x + y;
    }
}

int main() {
    int n;  // Change n to compute up to nth Fibonacci number
    int result;

    printf("Enter a number: ");
    scanf("%d",&n);

    #pragma omp parallel
    {
        #pragma omp single
        result = fib(n);
    }

    printf("Fibonacci number F(%d) = %d\n", n, result);
    return 0;
}
