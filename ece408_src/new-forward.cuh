#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *k, float *x, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K){

  __shared__ float MdA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float MdB[TILE_WIDTH][TILE_WIDTH];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, bz = blockIdx.z;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  int numAColumns = C*K*K;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  float pValue = 0.0;

  for(int ph=0; ph < ceil(numAColumns*1.0/TILE_WIDTH); ph++){
    
    int temp_x = ph * TILE_WIDTH + tx;
    int temp_y = ph * TILE_WIDTH + ty;

    int k_c = temp_x / (K*K);
    int k_h = temp_x % (K*K) / K;
    int k_w = temp_x % (K*K) % K; 
    int k_m = row;

    if(k_m < M && k_c < C){
      MdA[ty][tx] = k4d(k_m, k_c, k_h, k_w);
    }
    else{
      MdA[ty][tx] = 0.0;
    }

    int x_b = bz;
    int x_c = temp_y / (K*K);
    int x_h = col / W_out;
    int x_w = col % W_out; 
    int x_p = temp_y % (K*K) / K;
    int x_q = temp_y % (K*K) % K;

    if(x_b < B && x_c < C && (x_h + x_p) < H && (x_w + x_q) < W){
      MdB[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
    }
    else{
      MdB[ty][tx] = 0.0;
    }
    __syncthreads();

    for(int k=0; k<TILE_WIDTH; k++){
        pValue += MdA[ty][k] * MdB[k][tx];
    }
    __syncthreads();
  }
  
  int y_b = bz;
  int y_m = row;
  int y_h = (bx*TILE_WIDTH + tx) / W_out;
  int y_w = (bx*TILE_WIDTH + tx) % W_out;

  if(y_b < B && y_m < M){
    y4d(y_b, y_m, y_h, y_w) = pValue;
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

    const int B = x.shape_[0];
    const int M = y.shape_[1]; //number of output feature maps
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int out_size = H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(ceil(out_size*1.0/TILE_WIDTH), ceil(M*1.0/TILE_WIDTH), B);
    matrixMultiplyShared<<<gridDim, blockDim>>>(w.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K);

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