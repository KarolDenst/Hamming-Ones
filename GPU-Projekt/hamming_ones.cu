#include "hamming_ones.h"

#if defined (__INTELLISENSE__) | defined (__RESHARPER__)
template<class T1, class T2>
__device__ void atomicAdd(T1 x, T2 y);
#endif

int main()
{
    unsigned int* sequences = (unsigned int*)malloc(NUMBER * LENGTH * sizeof(unsigned int));
    GenerateSequences(sequences);
    unsigned int* result = (unsigned int*)malloc(sizeof(unsigned int));

    if (FORCE_ONE_PAIR) {
        CreateOnePair(sequences);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    if (CPU) {
        GetHammingOnes(sequences, result);
    }
    else {
        unsigned int* d_sequences;
        unsigned int* d_result;
        cudaMalloc(&d_sequences, NUMBER * LENGTH * sizeof(unsigned int));
        cudaMalloc(&d_result, sizeof(unsigned int));
        cudaMemcpy(d_sequences, sequences, NUMBER * LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, result, sizeof(unsigned int), cudaMemcpyHostToDevice);
        unsigned int blocks = (NUMBER + THREAD_COUNT - 1) / THREAD_COUNT;

        if (HASH) {
            int* keys = (int*)malloc(HASH_MAP_SIZE * sizeof(int));
            unsigned int* values = (unsigned int*)malloc(HASH_MAP_SIZE * sizeof(unsigned int) * LENGTH);
            int* d_keys;
            unsigned int* d_values;

            SetUpHashTable(keys, values, sequences);
            cudaMalloc(&d_keys, HASH_MAP_SIZE * sizeof(int));
            cudaMalloc(&d_values, HASH_MAP_SIZE * sizeof(unsigned int) * LENGTH);
            cudaMemcpy(d_keys, keys, HASH_MAP_SIZE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_values, values, HASH_MAP_SIZE * sizeof(unsigned int) * LENGTH, cudaMemcpyHostToDevice);

            GetHammingOnesGPUHash << <blocks * LENGTH, THREAD_COUNT >> > (d_sequences, d_result, d_keys, d_values);
            cudaFree(d_keys);
            cudaFree(d_values);
            free(keys);
            free(values);
        }
        else {
            GetHammingOnesGPU << <blocks, THREAD_COUNT >> > (d_sequences, d_result);
        }

        cudaMemcpy(result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_sequences);
        cudaFree(d_result);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    float seconds = (float)(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()) / 1000000.0;
        
    PrintResults(result[0], seconds);

    free(sequences);
    free(result);
    return 0;
}

void CreateOnePair(unsigned int* sequences) {
    for (int i = 0; i < LENGTH; i++) {
        sequences[69 * LENGTH + i] = sequences[420 * LENGTH + i];
    }
    sequences[69 * LENGTH] ^= 1UL << 4;
}

void GetHammingOnes(unsigned int* sequences, unsigned int* result) {
    result[0] = 0;
    for (int i = 0; i < NUMBER; i++) {
        for (int j = i + 1; j < NUMBER; j++) {
            if (CheckIfHammingOnes(&sequences[i * LENGTH], &sequences[j * LENGTH])) {
                PrintPair(&sequences[i * LENGTH], i, &sequences[j * LENGTH], j);
                result[0]++;
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
    srand(SEED);

    for (int i = 0; i < NUMBER * LENGTH; i++) {
        GenerateRandomBits(&sequences[i]);
    }
}

__host__ __device__ unsigned int CountSetBits(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

__host__ __device__ bool CheckIfHammingOnes(unsigned int* s1, unsigned int* s2) {
    int counter = 0;
    for (int i = 0; i < LENGTH; i++) {
        unsigned int xor = s1[i] ^ s2[i];
        counter += CountSetBits(xor);

        if (counter > 1) return false;
    }

    if (counter == 1) return true;
    return false;
}

__host__ __device__ void PrintBits(unsigned int num) {
    int size = sizeof(unsigned int);
    unsigned int maxPow = 1 << (size * 8 - 1);
    for (int i = 0; i < size; ++i) {
        for (; i < size * 8; ++i) {
            // print last bit and shift left.
            printf("%u ", num & maxPow ? 1 : 0);
            num = num << 1;
        }
    }
}

__host__ __device__ void PrintSequence(unsigned int* sequence) {
    for (int i = 0; i < LENGTH; i++) {
        PrintBits(sequence[i]);
    }
}

__host__ __device__ void PrintPair(unsigned int* s1, int i, unsigned int* s2, int j) {
    printf("%7d: ", i);
    PrintSequence(s1);
    printf("\n");
    printf("%7d: ", j);
    PrintSequence(s2);
    printf("\n==========\n");
}

unsigned int CountResultNumber(unsigned int* result) {
    unsigned int sum = 0;
    for (int i = 0; i < NUMBER; i++) {
        sum += result[i];
    }

    return sum;
}

void PrintResults(unsigned int result_number, float seconds) {
    printf("\n");
    printf("The program found %d results. It took %f seconds", result_number, seconds);
}

__global__ void GetHammingOnesGPU(unsigned int* sequences, unsigned int* result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= NUMBER) return;
    unsigned int* main = &sequences[id * LENGTH];
    for (int i = id + 1; i < NUMBER; i++) {
        if (CheckIfHammingOnes(main, &sequences[i * LENGTH])) {
            printf("%d - %d\n", id, i);
            atomicAdd(&result[0], 1);
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

__device__ bool CheckIfHammingZerosGPU(unsigned int* s1, unsigned int* s2) {
    for (int i = 0; i < LENGTH; i++) {
        unsigned int xor = s1[i] ^ s2[i];
        if (CountSetBitsGPU(xor) > 0) return false;
    }

    return true;
}

__global__ void GetHammingOnesGPUHash(unsigned int* sequences, unsigned int* result, int* keys, unsigned int* values) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= NUMBER * LENGTH) return;

    int i = id % LENGTH;
    int index = id - i;
    if(id == 0) result[0] = 0;
    unsigned int seq[LENGTH];

    for (int j = 0; j < LENGTH; j++) {
        seq[j] = sequences[index + j];
    }

    for (int j = 0; j < 32; j++) {
        seq[i] ^= 1UL << j;
        int key = GetKey(keys, values, seq);
                
        if (key != -1) {
            if (key < id / LENGTH) {
                printf("%d - %d\n", id / LENGTH, key);
                atomicAdd(&result[0], 1);
            }
        }
        seq[i] ^= 1UL << j;
    }
}

void Add(int* keys, unsigned int* values, int key, unsigned int* seq) {
    unsigned int i = HashSequence(seq);
    while (keys[i] != 0) {
        i = (i + 1) % HASH_MAP_SIZE;
    }

    keys[i] = key;
    for (int j = 0; j < LENGTH; j++) {
        values[i * LENGTH + j] = seq[j];
    }
}

__host__ __device__ unsigned int HashSequence(unsigned int* seq) {
    unsigned int result = 0;

    for (int i = 0; i < LENGTH; i++) {
        result = result ^ Hash(seq[i]);
    }

    return result % HASH_MAP_SIZE;
}

__host__ __device__ unsigned int Hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;

    return x;
}

void SetUpHashTable(int* keys, unsigned int* values, unsigned int* sequences) {
    SetUpKeys(keys);
    for (int i = 0; i < NUMBER; i++) {
        Add(keys, values, i, &sequences[i * LENGTH]);
    }
}

void SetUpKeys(int* keys) {
    for (int i = 0; i < HASH_MAP_SIZE; i++) {
        keys[i] = 0;
    }
}

__device__ int GetKey(int* keys, unsigned int* values, unsigned int* sequence) {
    int i = HashSequence(sequence);
    
    while (keys[i] != 0) {
        if (CheckIfHammingZerosGPU(&values[i * LENGTH], sequence)) return keys[i];

        i = (i + 1) % HASH_MAP_SIZE;
    }

    return -1;
}