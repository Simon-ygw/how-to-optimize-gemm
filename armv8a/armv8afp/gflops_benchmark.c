#include <time.h>
#include <stdio.h>

#define LOOP (1e9)
#define OP_FLOATS (80)

void TEST(int);

static double get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static double get_time_ns(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1e9 + (end->tv_nsec - start->tv_nsec);
}

void TestLoad_L1(void)
{
    float *data = (float*)malloc(16 * 1024);
    float *dataAlign = (float*)(((unsigned long long)data + 15) & ~(0xf));
    int loops = 1000000000;
    int outLoops = 1024;

    int i = 0;
    for (; i < 4096; i++) {
        dataAlign[i] = i;
    }

    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    asm volatile(    
        "2:                          \n"
        "ld1 {v13.4s}, [%0]          \n"
        "ld1 {v14.4s}, [%0]          \n"
        "ld1 {v15.4s}, [%0]          \n"
        "ld1 {v16.4s}, [%0]          \n"
        "ld1 {v17.4s}, [%0]          \n"
        "ld1 {v18.4s}, [%0]          \n"
        "ld1 {v19.4s}, [%0]          \n"
        "ld1 {v20.4s}, [%0]          \n"
        "ld1 {v21.4s}, [%0]          \n"
        "ld1 {v22.4s}, [%0]          \n"
        "ld1 {v23.4s}, [%0]          \n"
        "ld1 {v24.4s}, [%0]          \n"
        "ld1 {v25.4s}, [%0]          \n"
        "ld1 {v26.4s}, [%0]          \n"
        "ld1 {v27.4s}, [%0]          \n"
        "ld1 {v28.4s}, [%0]          \n"
        "subs %1, %1, #1             \n"
        "bgt 2b                      \n"
        : "=r"(dataAlign), "=r"(loops)
        : "0"(dataAlign), "1"(loops)
                : "memory", "cc", "x8", "x9", "x10", "x13", "x14", "x15", "v0", "v1", "v13", "v14","v15", "v16",
                    "v17","v18", "v9","v10","v11","v12", "v20", "v21", "v22", "v23");

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);

    printf("time_used = %.6lf, per cmd cost %.6lf ns\n", time_used, time_used / 16);
    free(data);
}

void TestFMLA(void)
{
    int loops = 1000000000;
    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    asm volatile(    
        "3:                          \n"
        "   fmla v16.4s, v2.4s, v0.s[0]      \n"
        "   fmla v20.4s, v2.4s, v0.s[1]      \n"
        "   fmla v24.4s, v2.4s, v0.s[2]      \n"
        "   fmla v28.4s, v2.4s, v0.s[3]      \n"
        "   fmla v17.4s, v3.4s, v0.s[0]      \n"
        "   fmla v21.4s, v3.4s, v0.s[1]      \n"
        "   fmla v25.4s, v3.4s, v0.s[2]      \n"
        "   fmla v29.4s, v3.4s, v0.s[3]      \n"
        "   fmla v18.4s, v4.4s, v0.s[0]      \n"
        "   fmla v22.4s, v4.4s, v0.s[1]      \n"
        "   fmla v26.4s, v4.4s, v0.s[2]      \n"
        "   fmla v30.4s, v4.4s, v0.s[3]      \n"
        "   fmla v19.4s, v5.4s, v0.s[0]      \n"      
        "   fmla v23.4s, v5.4s, v0.s[1]      \n"    
        "   fmla v27.4s, v5.4s, v0.s[2]      \n"  
        "   fmla v31.4s, v5.4s, v0.s[3]      \n" 
        "subs %1, %1, #1             \n"
        "bgt 3b                      \n"
        : "=r"(loops)
        : "0"(loops)
                : "memory", "cc", "x8", "x9", "x10", "x13", "x14", "x15", "v0", "v1", "v13", "v14","v15", "v16",
                    "v17","v18", "v9","v10","v11","v12", "v20", "v21", "v22", "v23");

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time_ns(&start, &end);

    printf("time_used = %.6lf\n", time_used);
}


void TestLoad_L2(void) 
{
    float *data0 = (float*)malloc(256 * 1024 + 16);
    float *dataPack0 = (float*)malloc(256 * 1024 + 16);

    float *data = (float*)(((unsigned long long)data0 + 15) & ~(0xf));

    float *dataPack = (float*)(((unsigned long long)dataPack0 + 15) & ~(0xf));

    for (int i = 0; i < 64 * 1024; i++) {
        dataPack[i] = data[i];
    }
    int loops = 1024;
    int outLoops = 1024;

    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    asm volatile(
        "4: \n"    
        "1:                               \n"
        "ld1 {v13.4s}, [%0], #16          \n"
        "ld1 {v14.4s}, [%0], #16          \n"
        "ld1 {v15.4s}, [%0], #16          \n"
        "ld1 {v16.4s}, [%0], #16          \n"
        "ld1 {v17.4s}, [%0], #16          \n"
        "ld1 {v18.4s}, [%0], #16          \n"
        "ld1 {v19.4s}, [%0], #16          \n"
        "ld1 {v20.4s}, [%0], #16          \n"
        "ld1 {v21.4s}, [%0], #16          \n"
        "ld1 {v22.4s}, [%0], #16          \n"
        "ld1 {v23.4s}, [%0], #16          \n"
        "ld1 {v24.4s}, [%0], #16          \n"
        "ld1 {v25.4s}, [%0], #16          \n"
        "ld1 {v26.4s}, [%0], #16          \n"
        "ld1 {v27.4s}, [%0], #16          \n"
        "ld1 {v28.4s}, [%0], #16          \n"
        "subs %1, %1, #1             \n"
        "bgt 1b                      \n"
        "subs %0, %0,  #256  \n"
        "subs %2, %2, #1  \n"  
        "bgt 4b \n"
        : "=r"(dataPack), "=r"(loops), "=r"(outLoops)
        : "0"(dataPack), "1"(loops), "2"(outLoops)
                : "memory", "cc", "x8", "x9", "x10", "x13", "x14", "x15", "v0", "v1", "v13", "v14","v15", "v16",
                    "v17","v18", "v9","v10","v11","v12", "v20", "v21", "v22", "v23");

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time_ns(&start, &end);

    printf("time_used = %.6lf\n", time_used);

    free(data0);
    free(dataPack0);
}

void test_gfops()
{
    struct timespec start, end;
    double time_used = 0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    TEST(LOOP);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);

    printf("perf: %.6lf %.6lf\r\n", LOOP * OP_FLOATS * 1.0 * 1e-9 / time_used, time_used);
}


int main()
{
    TestLoad_L1();
    TestLoad_L2();
    TestFMLA();
    test_gfops();
    return 0;
}