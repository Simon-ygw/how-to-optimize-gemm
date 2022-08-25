aarch64-linux-gnu-as -o test.o test.S
aarch64-linux-gnu-gcc -c gflops_benchmark.c
aarch64-linux-gnu-gcc -o gflops_benchmark gflops_benchmark.o test.o