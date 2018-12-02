#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16

#include <mxnet/base.h>

__constant__ float kernel[12][7][7];

namespace mxnet
{
namespace op
{

__global__ void forward_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K){
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int X_tile_width = TILE_WIDTH + K - 1; 

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;

    int m = blockIdx.x;
    int h = h_base + h0;
    int w = w_base + w0;
    int b = blockIdx.z;

    float acc = 0.0;
    for (int c = 0; c < C; c++){

        for(int i=h; i < h_base + X_tile_width; i+= TILE_WIDTH){
            for(int j=w; j < w_base + X_tile_width; j+= TILE_WIDTH)
                if(i < H && j < W){
                    X_shared[(i-h_base) * X_tile_width + j-w_base] = x4d(b,c,i,j);  
                }
                else{
                    X_shared[(i-h_base) * X_tile_width + j-w_base] = 0.0;
                }
        }
        __syncthreads();

        acc += X_shared[(h0+0) * X_tile_width + w0+0] * kernel[m][0][0]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+1] * kernel[m][0][1];         
        acc += X_shared[(h0+0) * X_tile_width + w0+2] * kernel[m][0][2];           
        acc += X_shared[(h0+0) * X_tile_width + w0+3] * kernel[m][0][3];         
        acc += X_shared[(h0+0) * X_tile_width + w0+4] * kernel[m][0][4];          
        acc += X_shared[(h0+0) * X_tile_width + w0+5] * kernel[m][0][5];          
        acc += X_shared[(h0+0) * X_tile_width + w0+6] * kernel[m][0][6];   

        acc += X_shared[(h0+1) * X_tile_width + w0+0] * kernel[m][1][0]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+1] * kernel[m][1][1];         
        acc += X_shared[(h0+1) * X_tile_width + w0+2] * kernel[m][1][2];           
        acc += X_shared[(h0+1) * X_tile_width + w0+3] * kernel[m][1][3];         
        acc += X_shared[(h0+1) * X_tile_width + w0+4] * kernel[m][1][4];          
        acc += X_shared[(h0+1) * X_tile_width + w0+5] * kernel[m][1][5];          
        acc += X_shared[(h0+1) * X_tile_width + w0+6] * kernel[m][1][6];   

        acc += X_shared[(h0+2) * X_tile_width + w0+0] * kernel[m][2][0]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+1] * kernel[m][2][1];         
        acc += X_shared[(h0+2) * X_tile_width + w0+2] * kernel[m][2][2];           
        acc += X_shared[(h0+2) * X_tile_width + w0+3] * kernel[m][2][3];         
        acc += X_shared[(h0+2) * X_tile_width + w0+4] * kernel[m][2][4];          
        acc += X_shared[(h0+2) * X_tile_width + w0+5] * kernel[m][2][5];          
        acc += X_shared[(h0+2) * X_tile_width + w0+6] * kernel[m][2][6];   

        acc += X_shared[(h0+3) * X_tile_width + w0+0] * kernel[m][3][0]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+1] * kernel[m][3][1];         
        acc += X_shared[(h0+3) * X_tile_width + w0+2] * kernel[m][3][2];           
        acc += X_shared[(h0+3) * X_tile_width + w0+3] * kernel[m][3][3];         
        acc += X_shared[(h0+3) * X_tile_width + w0+4] * kernel[m][3][4];          
        acc += X_shared[(h0+3) * X_tile_width + w0+5] * kernel[m][3][5];          
        acc += X_shared[(h0+3) * X_tile_width + w0+6] * kernel[m][3][6];   

        acc += X_shared[(h0+4) * X_tile_width + w0+0] * kernel[m][4][0]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+1] * kernel[m][4][1];         
        acc += X_shared[(h0+4) * X_tile_width + w0+2] * kernel[m][4][2];           
        acc += X_shared[(h0+4) * X_tile_width + w0+3] * kernel[m][4][3];         
        acc += X_shared[(h0+4) * X_tile_width + w0+4] * kernel[m][4][4];          
        acc += X_shared[(h0+4) * X_tile_width + w0+5] * kernel[m][4][5];          
        acc += X_shared[(h0+4) * X_tile_width + w0+6] * kernel[m][4][6];   

        acc += X_shared[(h0+5) * X_tile_width + w0+0] * kernel[m][5][0]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+1] * kernel[m][5][1];         
        acc += X_shared[(h0+5) * X_tile_width + w0+2] * kernel[m][5][2];           
        acc += X_shared[(h0+5) * X_tile_width + w0+3] * kernel[m][5][3];         
        acc += X_shared[(h0+5) * X_tile_width + w0+4] * kernel[m][5][4];          
        acc += X_shared[(h0+5) * X_tile_width + w0+5] * kernel[m][5][5];          
        acc += X_shared[(h0+5) * X_tile_width + w0+6] * kernel[m][5][6];   

        acc += X_shared[(h0+6) * X_tile_width + w0+0] * kernel[m][6][0]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+1] * kernel[m][6][1];         
        acc += X_shared[(h0+6) * X_tile_width + w0+2] * kernel[m][6][2];           
        acc += X_shared[(h0+6) * X_tile_width + w0+3] * kernel[m][6][3];         
        acc += X_shared[(h0+6) * X_tile_width + w0+4] * kernel[m][6][4];          
        acc += X_shared[(h0+6) * X_tile_width + w0+5] * kernel[m][6][5];          
        acc += X_shared[(h0+6) * X_tile_width + w0+6] * kernel[m][6][6];  

        __syncthreads();
    }

    if(h < H_out && w < W_out){
        y4d(b, m, h, w) = acc;
    }   

    #undef y4d
    #undef x4d
    #undef k4d
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int X_tile_width = TILE_WIDTH + K - 1; //The X-tile width indicated on the manual

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* kernel = &shmem[X_tile_width * X_tile_width];

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;

    int m = blockIdx.x;
    int h = h_base + h0;
    int w = w_base + w0;
    int b = blockIdx.z;

    float acc = 0.0;
    for (int c = 0; c < C; c++){

        if(h0 < K && w0 < K){
            kernel[h0 * K + w0] = k4d(m, c, h0, w0);
        }   
        
        for(int i=h; i < h_base + X_tile_width; i+= TILE_WIDTH){
            for(int j=w; j < w_base + X_tile_width; j+= TILE_WIDTH)
                if(i < H && j < W){
                    X_shared[(i-h_base) * X_tile_width + j-w_base] = x4d(b,c,i,j);  
                }
                else{
                    X_shared[(i-h_base) * X_tile_width + j-w_base] = 0.0;
                }
        }
        __syncthreads();

        acc += X_shared[(h0+0) * X_tile_width + w0+0] * kernel[(0) * K + 0]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+1] * kernel[(0) * K + 1]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+2] * kernel[(0) * K + 2]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+3] * kernel[(0) * K + 3]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+4] * kernel[(0) * K + 4]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+5] * kernel[(0) * K + 5]; 
        acc += X_shared[(h0+0) * X_tile_width + w0+6] * kernel[(0) * K + 6]; 

        acc += X_shared[(h0+1) * X_tile_width + w0+0] * kernel[(1) * K + 0]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+1] * kernel[(1) * K + 1]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+2] * kernel[(1) * K + 2]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+3] * kernel[(1) * K + 3]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+4] * kernel[(1) * K + 4]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+5] * kernel[(1) * K + 5]; 
        acc += X_shared[(h0+1) * X_tile_width + w0+6] * kernel[(1) * K + 6];

        acc += X_shared[(h0+2) * X_tile_width + w0+0] * kernel[(2) * K + 0]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+1] * kernel[(2) * K + 1]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+2] * kernel[(2) * K + 2]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+3] * kernel[(2) * K + 3]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+4] * kernel[(2) * K + 4]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+5] * kernel[(2) * K + 5]; 
        acc += X_shared[(h0+2) * X_tile_width + w0+6] * kernel[(2) * K + 6]; 

        acc += X_shared[(h0+3) * X_tile_width + w0+0] * kernel[(3) * K + 0]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+1] * kernel[(3) * K + 1]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+2] * kernel[(3) * K + 2]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+3] * kernel[(3) * K + 3]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+4] * kernel[(3) * K + 4]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+5] * kernel[(3) * K + 5]; 
        acc += X_shared[(h0+3) * X_tile_width + w0+6] * kernel[(3) * K + 6]; 

        acc += X_shared[(h0+4) * X_tile_width + w0+0] * kernel[(4) * K + 0]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+1] * kernel[(4) * K + 1]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+2] * kernel[(4) * K + 2]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+3] * kernel[(4) * K + 3]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+4] * kernel[(4) * K + 4]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+5] * kernel[(4) * K + 5]; 
        acc += X_shared[(h0+4) * X_tile_width + w0+6] * kernel[(4) * K + 6]; 

        acc += X_shared[(h0+5) * X_tile_width + w0+0] * kernel[(5) * K + 0]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+1] * kernel[(5) * K + 1]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+2] * kernel[(5) * K + 2]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+3] * kernel[(5) * K + 3]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+4] * kernel[(5) * K + 4]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+5] * kernel[(5) * K + 5]; 
        acc += X_shared[(h0+5) * X_tile_width + w0+6] * kernel[(5) * K + 6]; 

        acc += X_shared[(h0+6) * X_tile_width + w0+0] * kernel[(6) * K + 0]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+1] * kernel[(6) * K + 1]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+2] * kernel[(6) * K + 2]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+3] * kernel[(6) * K + 3]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+4] * kernel[(6) * K + 4]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+5] * kernel[(6) * K + 5]; 
        acc += X_shared[(h0+6) * X_tile_width + w0+6] * kernel[(6) * K + 6]; 

        __syncthreads();
    }

    if(h < H_out && w < W_out){
        y4d(b, m, h, w) = acc;
    }   

    #undef y4d
    #undef x4d
    #undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));
    int Total_grid = W_grid * H_grid;

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Total_grid, B);
    size_t shmem_size;
    
    //cudaMemcpyToSymbol(kernel, w, C*M*K*K*sizeof(float));
    if(C == 1){
        cudaMemcpyToSymbol(kernel, w.dptr_, M*K*K*sizeof(float));
        shmem_size = sizeof(float) * ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1)); 
        forward_constant<<<gridDim,blockDim,shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
    }
    else{
        shmem_size = sizeof(float) * ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K); 
        forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
    }
    
    // Call the kernel
    //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif