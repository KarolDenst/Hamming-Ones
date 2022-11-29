#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <random>
#include <chrono>

#define CPU false
#define NUMBER 100000
#define LENGTH 1 // Sequence length = LENGTH * 32
#define THREAD_COUNT 1024

void GetHammingOnes(unsigned int* sequences, unsigned int* result);
void GenerateRandomBits(unsigned int* sequence);
void GenerateSequences(unsigned int* sequences);
unsigned int CountSetBits(unsigned int n);
bool CheckIfHammingOnes(unsigned int* s1, unsigned int* s2);
void PrintBits(unsigned int num);
void PrintSequence(unsigned int* sequence);
void PrintPair(unsigned int* s1, int i, unsigned int* s2, int j);
unsigned int CountResultNumber(unsigned int* result);
void PrintResults(unsigned int result_number, long long seconds);

__global__ void GetHammingOnesGPU(unsigned int* sequences, unsigned int* result);
__device__ unsigned int CountSetBitsGPU(unsigned int n);
__device__ bool CheckIfHammingOnesGPU(unsigned int* s1, unsigned int* s2);
__device__ void GetBitsGPU(unsigned int num);
__device__ void GetSequenceGPU(unsigned int* sequence);
__device__ void GetPairGPU(unsigned int* s1, int i, unsigned int* s2, int j);
