CC = cc 
VT = -g -dynamic
QOPT = -qopt-report=5
AVX = -mavx
OPT = -O3 -ftree-vectorize -funroll-loops
#OPT = -fno-alias 
#CFLAGS = -Wall -std=gnu99 $(OPT) $(AVX) $(QOPT)
CFLAGS = -Wall -std=gnu99 $(OPT) $(AVX)
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt

targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o  

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)