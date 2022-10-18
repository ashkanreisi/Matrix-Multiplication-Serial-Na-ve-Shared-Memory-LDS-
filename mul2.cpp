#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <ctime>



#define HIP_ASSERT(x) (assert((x)==hipSuccess))

// HIP kernel. Each thread takes care of one element of c
__global__ void Mul(float* MatA, float* VecB, float* VecC, int n, int m)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
   
    float sum = 0;

    // Make sure we do not go out of bounds
    if (id < n)
    {
        for (int i = 0; i < m; i++)
        {
            sum += MatA[id * m + i] * VecB[id];
        }
        VecC[id] = sum;
    }
}


__global__ void SharedMemoryMul(float *MatA, float *VecB, float *VecC,int n, int m)
{
    const int blockSize = 1024;
    __shared__ float row[blockSize];
    __shared__ float vecPartial[blockSize];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    double sum = 0.0;

    for (int tile_idx = 0; tile_idx < (m / blockSize); tile_idx++)
    {
        int subMat = idx * m + tile_idx * blockSize;
        row[idy] = MatA[subMat + idy];

        int subVec = tile_idx * blockSize;
        vecPartial[idy] = VecB[subVec + idy];

        __syncthreads();

        for (int i = 0; i < blockSize; i++)
        {
            sum += row[i] * vecPartial[i];
        }

        __syncthreads();
    }

    VecC[idx] = sum;
    return;
}


 
int main()
{
    // Size of vectors,matrix and their values
    const int n = 128, m = 128;
    int v1 = 1, v2 = 2;
 
    // Host input vectors
    float *h_MatA;
    float *h_VecB;
    //Host output vector
    float *h_VecC;
    //Host output vector for verification
    float *h_VerifyC;
 
    // Device input vectors
    float *d_MatA;
    float *d_VecB;
    //Device output vector
    float *d_VecC;
 
    // Size, in bytes, of each vector
    size_t bytes_MatA = n*m*sizeof(float);
    size_t bytes_VecB = m*sizeof(float);
    size_t bytes_VecC = n * sizeof(float);
    // Allocate memory for each vector on host
    h_MatA = (float*)malloc(bytes_MatA);
    h_VecB = (float*)malloc(bytes_VecB);
    h_VecC = (float*)malloc(bytes_VecC);
    h_VerifyC = (float*)malloc(bytes_VecC);

   printf("Finished allocating vectors on the CPU\n");     
    // Allocate memory for each vector on GPU
   HIP_ASSERT(hipMalloc(&d_MatA, bytes_MatA));
   HIP_ASSERT(hipMalloc(&d_VecB, bytes_VecB));
   HIP_ASSERT(hipMalloc(&d_VecC, bytes_VecC));
 
   printf("Finished allocating vectors on the GPU\n");

    // Initialize vectors on host
   for (int i = 0; i < n; i++) {
       for (int j = 0; j < m; j++) {
           if (n == 1024) {
               v1 = i; //every element takes the value of its i index in case of a Matrix of size 1024x1024.
           }
           if (n == 2048) {
               v1 = j;//every element takes the value of its j index in case of a Matrix of size 2048x2048.
           }
           h_MatA[i * m + j] = v1;
       }
   }

   for (int i = 0; i < m; i++) {
       h_VecB[i] = v2;
   }
 
    // Copy host vectors to device
    HIP_ASSERT(hipMemcpy( d_MatA, h_MatA, bytes_MatA, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_VecB, h_VecB, bytes_VecB, hipMemcpyHostToDevice));
 
    printf("Finished copying vectors to the GPU\n");

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil(float(n)/blockSize);
 
    printf("Launching the  kernel on the GPU\n");
    // Execute the kernel
    //hipLaunchKernelGGL(Mul, dim3(gridSize), dim3(blockSize), 0, 0, d_MatA, d_VecB, d_VecC, n,m);
    // Use HIP Events for timing
    hipEvent_t start, stop;
    float time;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
    Mul <<<gridSize, blockSize>>> (d_MatA, d_VecB, d_VecC, n, m);
    hipDeviceSynchronize( );

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time, start, stop);
    std::cout << " Hip Kernel Matrix Multiplication time =" << '\t' << time
        << "ms" << std::endl;
    printf("Finished executing kernel\n");
    // Copy array back to host
   HIP_ASSERT(hipMemcpy( h_VecC, d_VecC, bytes_VecC, hipMemcpyDeviceToHost));
   printf("Finished copying the output vector from the GPU to the CPU\n");


   


   //Compute for CPU 
   //Using Ctime for timing
   std::clock_t cStart = std::clock();
     for (int i=0;i<n;i++){
         h_VerifyC[i]=0;
        for (int j=0;j<m;j++){
             h_VerifyC[i]= h_VerifyC[i]+h_MatA[i * m + j]*h_VecB[j];
        }
     }
   std::clock_t cEnd = std::clock();
   double timeElapsed_ms;
   timeElapsed_ms = 1000.0 * (cEnd - cStart) / CLOCKS_PER_SEC;
   std::cout << " CPU Matrix Multiplication time =" << '\t' << timeElapsed_ms
       << "ms" << std::endl;
  


    //Verfiy results
    for(int i=0; i <n; i++)
    {
    if (abs(h_VerifyC[i] - h_VecC[i]) > 1e-5) 
     {
     printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_VecC[i], d_VecC[i]);
     }  
    }	
    printf("\n");
    printf("\n");
    printf("Printing few elements from the output vector\n");

    for(int i=0; i < 20; i++)
    {
     printf("Output[%d]:%f\n",i, h_VecC[i]);	    
    }

    printf("Releasing GPU memory\n");
  



   
    printf("\n");
    printf("\n");
    // Release device memory
    HIP_ASSERT(hipFree(d_MatA));
    HIP_ASSERT(hipFree(d_VecB));
    HIP_ASSERT(hipFree(d_VecC));
 
    // Release host memory
    printf("Releasing CPU memory\n");
    free(h_MatA);
    free(h_VecB);
    free(h_VecC);


    //Execute the kernel for LDS
    // Allocate memory for each vector on host
    h_MatA = (float*)malloc(bytes_MatA);
    h_VecB = (float*)malloc(bytes_VecB);
    h_VecC = (float*)malloc(bytes_VecC);
    h_VerifyC = (float*)malloc(bytes_VecC);

   printf("Finished allocating vectors on the CPU\n");     
    // Allocate memory for each vector on GPU
   HIP_ASSERT(hipMalloc(&d_MatA, bytes_MatA));
   HIP_ASSERT(hipMalloc(&d_VecB, bytes_VecB));
   HIP_ASSERT(hipMalloc(&d_VecC, bytes_VecC));
 
   printf("Finished allocating vectors on the GPU\n");

   // Copy host vectors to device
    HIP_ASSERT(hipMemcpy( d_MatA, h_MatA, bytes_MatA, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_VecB, h_VecB, bytes_VecB, hipMemcpyHostToDevice));
 
    printf("Finished copying vectors to the GPU\n");

    //int gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil(float(n)/blockSize);
     
    printf("Launching the  kernel on the GPU\n");
     hipEvent_t start2, stop2;
    float time2;
    hipEventCreate(&start2);
    hipEventCreate(&stop2);
    hipEventRecord(start2, 0);
    SharedMemoryMul <<<gridSize, blockSize>>> (d_MatA, d_VecB, d_VecC, n, m);
    hipDeviceSynchronize( );

    hipEventRecord(stop2, 0);
    hipEventSynchronize(stop2);
    hipEventElapsedTime(&time2, start2, stop2);
    std::cout << " Hip Kernel Shared Memory Matrix Multiplication time =" << '\t' << time2
        << "ms" << std::endl;
    printf("Finished executing kernel\n");
    // Copy array back to host
   HIP_ASSERT(hipMemcpy( h_VecC, d_VecC, bytes_VecC, hipMemcpyDeviceToHost));
   printf("Finished copying the output vector from the GPU to the CPU\n");

       // Release device memory
    HIP_ASSERT(hipFree(d_MatA));
    HIP_ASSERT(hipFree(d_VecB));
    HIP_ASSERT(hipFree(d_VecC));
 
    // Release host memory
    printf("Releasing CPU memory\n");
    free(h_MatA);
    free(h_VecB);
    free(h_VecC);


 
    return 0;
}
