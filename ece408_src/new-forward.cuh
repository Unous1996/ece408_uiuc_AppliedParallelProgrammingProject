
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define BLOCK_SIZE_Z 4

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
        /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    __shared__ float T[2*BLOCK_SIZE_Z]; 

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));

    int m = blockIdx.x;
    int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;

    int stride;
    int tz = threadIdx.z;

    float acc = 0;

    for (int c = 0; c < C; c++){
        for (int p = 0; p < K; p++){
            stride = 1;
            float temp = 0.0;
            if(tz < K){
                T[tz] = x4d(b, c, h + p, w + tz) * k4d(m, c, p, tz);
            }
            else{
                T[tz] = 0.0;
            }

            if(tz + 4 < K){
                T[tz + 4] = x4d(b, c, h + p, w + tz + 4) * k4d(m, c, p, tz + 4);
            }
            else{
                T[tz + 4] = 0.0;
            }

            /*
            if(b == 0 && c == 0 && h == 0 && w == 0 && p == 0){
                printf("Load check");
                for(int i_t = 0; i_t < 7; i_t++){
                    printf("T[%d] = %f \n" ,i_t, T[i_t]);
                    printf("Where the value should be loaded is %f \n", x4d(b, c, h + p, w + i_t) * k4d(m, c, p, i_t));
                }
            }
            */

            while(stride < 8){
                __syncthreads();
                int index = (tz + 1) * stride * 2 - 1;
                if(index < 2*BLOCK_SIZE_Z && index - stride >= 0){
                    T[index] += T[index - stride];
                }
                stride = stride * 2;
            }

            //printf("Result check");
            for(int q = 0; q < K; q++){
                temp += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                /*
                if(b == 0 && c == 0 && h == 0 && w == 0 && p == 0){
                    printf("T[%d] = %f \n" , q, T[q]);
                    printf("Correct[%d] = %f\n", q, temp);
                }
                */
            }

            acc += T[7];
        }
    }

    if (h < H_out && w < W_out) {
        y4d(b, m, h, w) = acc;
    }   

    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

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

    printf("K = %d\n", K);
    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, BLOCK_SIZE_Z);
    dim3 gridDim(M, Total_grid, B);

    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);

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