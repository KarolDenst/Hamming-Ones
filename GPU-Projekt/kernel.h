#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <random>
#include <chrono>

#define CPU false
#define HASH true
#define NUMBER 100000
#define LENGTH 32 // Sequence length = LENGTH * 32
#define THREAD_COUNT 1024
#define HASH_MAP_SIZE 200000

void GetHammingOnes(unsigned int* sequences, unsigned int* result);
void GenerateRandomBits(unsigned int* sequence);
void GenerateSequences(unsigned int* sequences);
__host__ __device__ unsigned int CountSetBits(unsigned int n);
__host__ __device__ bool CheckIfHammingOnes(unsigned int* s1, unsigned int* s2);
__host__ __device__ void PrintBits(unsigned int num);
__host__ __device__ void PrintSequence(unsigned int* sequence);
__host__ __device__ void PrintPair(unsigned int* s1, int i, unsigned int* s2, int j);
unsigned int CountResultNumber(unsigned int* result);
void PrintResults(unsigned int result_number, float seconds);

__global__ void GetHammingOnesGPU(unsigned int* sequences, unsigned int* result);
__device__ unsigned int CountSetBitsGPU(unsigned int n);

void Add(int* keys, unsigned int* values, int key, unsigned int* seq);
void SetUpHashTable(int* keys, unsigned int* values, unsigned int* sequences);
void SetUpKeys(int* keys);
__host__ __device__ unsigned int HashSequence(unsigned int* seq);
__host__ __device__ unsigned int Hash(unsigned int x);
__global__ void GetHammingOnesGPUHash(unsigned int* sequences, unsigned int* result, int* keys, unsigned int* values);
__device__ bool HasKey(int* keys, unsigned int* values, unsigned int* sequence);
__device__ bool CheckIfHammingZerosGPU(unsigned int* s1, unsigned int* s2);
