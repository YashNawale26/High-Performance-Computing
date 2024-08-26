#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

#define BUFFER_SIZE 10
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 2
#define NUM_ITEMS 40

int buffer[BUFFER_SIZE];
int in = 0;
int out = 0;
int count = 0;

omp_lock_t mutex;
omp_lock_t full;
omp_lock_t empty;

void init_locks() {
    omp_init_lock(&mutex);
    omp_init_lock(&full);
    omp_init_lock(&empty);
    omp_set_lock(&empty);  // Buffer is initially empty
}

void destroy_locks() {
    omp_destroy_lock(&mutex);
    omp_destroy_lock(&full);
    omp_destroy_lock(&empty);
}

void produce(int id) {
    int item = rand() % 100;  // Generate a random item
    omp_set_lock(&empty);
    omp_set_lock(&mutex);

    buffer[in] = item;
    in = (in + 1) % BUFFER_SIZE;
    count++;

    printf("Producer %d produced item %d (count: %d)\n", id, item, count);

    omp_unset_lock(&mutex);
    if (count == 1) omp_unset_lock(&full);  // Buffer is no longer empty
}

int consume(int id) {
    int item;
    omp_set_lock(&full);
    omp_set_lock(&mutex);

    item = buffer[out];
    out = (out + 1) % BUFFER_SIZE;
    count--;

    printf("Consumer %d consumed item %d (count: %d)\n", id, item, count);

    omp_unset_lock(&mutex);
    if (count == BUFFER_SIZE - 1) omp_unset_lock(&empty);  // Buffer is no longer full

    return item;
}

void producer(int id) {
    for (int i = 0; i < NUM_ITEMS / NUM_PRODUCERS; i++) {
        produce(id);
        usleep(rand() % 100000);  // Sleep for up to 0.1 seconds
    }
}

void consumer(int id) {
    for (int i = 0; i < NUM_ITEMS / NUM_CONSUMERS; i++) {
        consume(id);
        usleep(rand() % 100000);  // Sleep for up to 0.1 seconds
    }
}

int main() {
    srand(time(NULL));
    init_locks();

    #pragma omp parallel sections num_threads(NUM_PRODUCERS + NUM_CONSUMERS)
    {
        #pragma omp section
        {
            producer(0);
        }
        #pragma omp section
        {
            producer(1);
        }
        #pragma omp section
        {
            consumer(0);
        }
        #pragma omp section
        {
            consumer(1);
        }
    }

    destroy_locks();
    return 0;
}