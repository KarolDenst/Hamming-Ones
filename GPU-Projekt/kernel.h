#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#define CPU false
#define NUMBER 100000
#define LENGTH 1 // Sequence length = LENGTH * 32
#define THREAD_COUNT 1024

void GetHammingOnes(unsigned int* sequences);
void GenerateRandomBits(unsigned int* sequence);
void GenerateSequences(unsigned int* sequences);
unsigned int CountSetBits(unsigned int n);
bool CheckIfHammingOnes(unsigned int* s1, unsigned int* s2);
void PrintBits(unsigned int num);
void PrintSequence(unsigned int* sequence);

__global__ void GetHammingOnesGPU(unsigned int* sequences, unsigned int* result);
__device__ unsigned int CountSetBitsGPU(unsigned int n);
__device__ bool CheckIfHammingOnesGPU(unsigned int* s1, unsigned int* s2);
