#include "kernel.h"

int main()
{
    unsigned int sequences[NUMBER][LENGTH];
    GenerateSequences(sequences);

    GetHammingOnesCPU(sequences);
    return 0;
}

void GetHammingOnesCPU(unsigned int sequences[NUMBER][LENGTH]) {
    for (int i = 0; i < NUMBER; i++) {
        for (int j = i + 1; j < NUMBER; j++) {
            if (checkIfHammingOnes(sequences[i], sequences[j])) {
                PrintSequence(sequences[i]);
                printf("\n");
                PrintSequence(sequences[j]);
                printf("\n==========\n");
            }
        }
    }
}

void GenerateRandomBits(unsigned int sequence[LENGTH]) {
    unsigned int* result = new unsigned int[LENGTH];
    for (int i = 0; i < LENGTH; i++) {
        unsigned int start = ((unsigned int)rand()) << 17;
        unsigned int middle = ((unsigned int)rand()) << 2;
        unsigned int end = rand() % 3;
        sequence[i] = start + middle + end;
    }
}

void GenerateSequences(unsigned int sequences[NUMBER][LENGTH]) {
    srand(1);

    for (int i = 0; i < NUMBER; i++) {
        GenerateRandomBits(sequences[i]);
    }
}

unsigned int countSetBits(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

bool checkIfHammingOnes(unsigned int* s1, unsigned int* s2) {
    int counter = 0;
    for (int i = 0; i < LENGTH; i++) {
        unsigned int xor = s1[i] ^ s2[i];
        counter += countSetBits(xor);

        if (counter > 1) return false;
    }

    if (counter == 1) return true;
    return false;
}

void printBits(unsigned int num) {
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
        printBits(sequence[i]);
    }
}
