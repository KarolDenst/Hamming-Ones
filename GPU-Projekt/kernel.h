#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

#define NUMBER 100000
#define LENGTH 1 // Sequence length = LENGTH * 32

void GetHammingOnesCPU(unsigned int[NUMBER][LENGTH]);

void GenerateRandomBits(unsigned int sequence[LENGTH]);

void GenerateSequences(unsigned int sequences[NUMBER][LENGTH]);

unsigned int countSetBits(unsigned int n);

bool checkIfHammingOnes(unsigned int* s1, unsigned int* s2);

void printBits(unsigned int num);

void PrintSequence(unsigned int* sequence);
