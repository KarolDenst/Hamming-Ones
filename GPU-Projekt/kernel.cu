#include "kernel.h"

int main()
{
    unsigned int* sequences = (unsigned int*)malloc(NUMBER * LENGTH * sizeof(unsigned int));
    GenerateSequences(sequences);
    unsigned int* result = (unsigned int*)malloc(NUMBER * sizeof(unsigned int));

    if (CPU) GetHammingOnes(sequences);
    else {
        unsigned int* d_sequences;
        unsigned int* d_result;
        cudaMalloc(&d_sequences, NUMBER * LENGTH * sizeof(unsigned int));
        cudaMalloc(&d_result, NUMBER * sizeof(unsigned int));
        cudaMemcpy(d_sequences, sequences, NUMBER * LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int blocks = NUMBER / THREAD_COUNT + 1;
        GetHammingOnesGPU<<<blocks, THREAD_COUNT>>>(d_sequences, d_result);
        cudaMemcpy(result, d_result, NUMBER * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    
    return 0;
}

void GetHammingOnes(unsigned int* sequences) {
    for (int i = 0; i < NUMBER; i++) {
        for (int j = i + 1; j < NUMBER; j++) {
            if (CheckIfHammingOnes(&sequences[i * LENGTH], &sequences[j * LENGTH])) {
                printf("%d: ", i);
                //PrintSequence(sequences[i]);
                printf("\n");
                printf("%d: ", j);
                //PrintSequence(sequences[j]);
                printf("\n==========\n");
            }
        }
    }
}

void GenerateRandomBits(unsigned int* sequence) {
    unsigned int start = ((unsigned int)rand()) << 17;
    unsigned int middle = ((unsigned int)rand()) << 2;
    unsigned int end = rand() % 3;
    *sequence = start + middle + end;
}

void GenerateSequences(unsigned int* sequences) {
    srand(1);

    for (int i = 0; i < NUMBER * LENGTH; i++) {
        GenerateRandomBits(&sequences[i]);
    }
}

unsigned int CountSetBits(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

bool CheckIfHammingOnes(unsigned int* s1, unsigned int* s2) {
    int counter = 0;
    for (int i = 0; i < LENGTH; i++) {
        unsigned int xor = s1[i] ^ s2[i];
        counter += CountSetBits(xor);

        if (counter > 1) return false;
    }

    if (counter == 1) return true;
    return false;
}

void PrintBits(unsigned int num) {
    int size = sizeof(unsigned int);
    unsigned int maxPow = 1 << (size * 8 - 1);
    int i = 0;
    for (; i < size; ++i) {
        for (; i < size * 8; ++i) {
            // print last bit and shift left.
            printf("%u ", num & maxPow ? 1 : 0);
            num = num << 1;
        }
    }
}

void PrintSequence(unsigned int* sequence) {
    for (int i = 0; i < LENGTH; i++) {
        PrintBits(sequence[i]);
    }
}

__global__ void GetHammingOnesGPU(unsigned int* sequences, unsigned int* result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    result[id] = 0;
    if (id >= NUMBER) return;
    unsigned int* main = &sequences[id * LENGTH];
    for (int i = 0; i < NUMBER; i++) {
        if (CheckIfHammingOnesGPU(main, &sequences[i * LENGTH])) {
            printf("test");
            result[id]++;
        }
    }
}

__device__ unsigned int CountSetBitsGPU(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

__device__ bool CheckIfHammingOnesGPU(unsigned int* s1, unsigned int* s2) {
    int counter = 0;
    for (int i = 0; i < LENGTH; i++) {
        unsigned int xor = s1[i] ^ s2[i];
        counter += CountSetBitsGPU(xor);

        if (counter > 1) return false;
    }

    if (counter == 1) return true;
    return false;
}
