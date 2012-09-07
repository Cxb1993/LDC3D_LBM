CU_CC=nvcc
CU_FLAGS=-O3 -arch compute_20 -Xcompiler -fopenmp
CU_LIBS=

CC=g++
CC_FLAGS=-O3
CC_LIBS=-fopenmp -lcudart -lstdc++ 
CC_CUL=/usr/local/cuda-5.0/lib64
CC_CUI=/usr/local/cuda-5.0/include


SOURCE=clbm_ldc3D_17july.cpp
TARGET=clbm_ldc3D

$(TARGET): $(SOURCE) lbm_utils.o vtk_lib.o lbm_kernels.o
	$(CC) -o $(TARGET) $(SOURCE) -L$(CC_CUL) -I$(CC_CUI) $(CC_LIBS) lbm_utils.o vtk_lib.o lbm_kernels.o

lbm_utils.o:  lbm_utils.cpp
	$(CC) -c lbm_utils.cpp $(CC_FLAGS) 

vtk_lib.o: vtk_lib.cxx
	$(CC) -c vtk_lib.cxx $(CC_FLAGS)

lbm_kernels.o: lbm_kernels.cu
	$(CU_CC) -c lbm_kernels.cu $(CU_FLAGS)


clean:
	rm *.o $(TARGET)


