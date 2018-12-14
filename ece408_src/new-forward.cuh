#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH_BIG 32
#define TILE_WIDTH_SMALL 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
  
__global__ void matrixMultiplyShared32(float *k, float *x, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K){

  __shared__ float MdA[TILE_WIDTH_BIG][TILE_WIDTH_BIG];
  __shared__ float MdB[TILE_WIDTH_BIG][TILE_WIDTH_BIG];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, bz = blockIdx.z;

  int row1 = by * TILE_WIDTH_BIG + 2*ty;
  int row2 = by * TILE_WIDTH_BIG + 2*ty + 1;
  
  int col1 = bx * TILE_WIDTH_BIG + 2*tx;
  int col2 = bx * TILE_WIDTH_BIG + 2*tx + 1;

  int numAColumns = C*K*K;
  int numCRows = M;
  int numCColumns = H_out * W_out;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  float pValue1 = 0.0;
  float pValue2 = 0.0;
  float pValue3 = 0.0;
  float pValue4 = 0.0;

  int max_phases = ceil(numAColumns*1.0/TILE_WIDTH_BIG);
  
  for(int ph=0; ph < max_phases; ph++){  

    int temp_y1 = ph * TILE_WIDTH_BIG + 2*ty;
    int temp_y2 = ph * TILE_WIDTH_BIG + 2*ty + 1;

    int temp_x1 = ph * TILE_WIDTH_BIG + 2*tx;
    int temp_x2 = ph * TILE_WIDTH_BIG + 2*tx + 1;

    int k_c1 = temp_x1 / (K*K);
    int k_h1 = temp_x1 % (K*K) / K;
    int k_w1 = temp_x1 % (K*K) % K; 
    int k_m1 = row1;

    int k_c2 = temp_x2 / (K*K);
    int k_h2 = temp_x2 % (K*K) / K;
    int k_w2 = temp_x2 % (K*K) % K; 
    int k_m2 = k_m1;

    int k_c3 = temp_x1 / (K*K);
    int k_h3 = temp_x1 % (K*K) / K;
    int k_w3 = temp_x1 % (K*K) % K; 
    int k_m3 = row2;

    int k_c4 = temp_x2 / (K*K);
    int k_h4 = temp_x2 % (K*K) / K;
    int k_w4 = temp_x2 % (K*K) % K; 
    int k_m4 = k_m3;

    if(k_m1 < M && k_c1 < C){
      MdA[2*ty][2*tx] = k4d(k_m1, k_c1, k_h1, k_w1);
    }
    else{
      MdA[2*ty][2*tx] = 0.0;
    }

    if(k_m2 < M && k_c2 < C){
      MdA[2*ty][2*tx+1] = k4d(k_m2, k_c2, k_h2, k_w2);
    }
    else{
      MdA[2*ty][2*tx+1] = 0.0;
    }

   if(k_m3 < M && k_c3 < C){
      MdA[2*ty+1][2*tx] = k4d(k_m3, k_c3, k_h3, k_w3);
    }
    else{
      MdA[2*ty+1][2*tx] = 0.0;
    }

   if(k_m4 < M && k_c4 < C){
      MdA[2*ty+1][2*tx+1] = k4d(k_m4, k_c4, k_h4, k_w4);
    }
    else{
      MdA[2*ty+1][2*tx+1] = 0.0;
    }

    int x_b1 = bz;
    int x_c1 = temp_y1 / (K*K);
    int x_h1 = col1 / W_out;
    int x_w1 = col1 % W_out; 
    int x_p1 = temp_y1 % (K*K) / K;
    int x_q1 = temp_y1 % K;

    int x_b2 = x_b1;
    int x_c2 = x_c1;
    int x_h2 = col2 / W_out;
    int x_w2 = col2 % W_out; 
    int x_p2 = x_p1;
    int x_q2 = x_q1;

    int x_b3 = bz;
    int x_c3 = temp_y2 / (K*K);
    int x_h3 = col1 / W_out;
    int x_w3 = col1 % W_out; 
    int x_p3 = temp_y2 % (K*K) / K;
    int x_q3 = temp_y2 % K;

    int x_b4 = x_b3;
    int x_c4 = x_c3;
    int x_h4 = col2 / W_out;
    int x_w4 = col2 % W_out; 
    int x_p4 = x_p3;
    int x_q4 = x_q3;

    if(x_b1 < B && x_c1 < C && (x_h1 + x_p1) < H && (x_w1 + x_q1) < W){
      MdB[2*ty][2*tx] = x4d(x_b1, x_c1, x_h1 + x_p1, x_w1 + x_q1);
    }
    else{
      MdB[2*ty][2*tx] = 0.0;
    }

    if(x_b2 < B && x_c2 < C && (x_h2 + x_p2) < H && (x_w2 + x_q2) < W){
      MdB[2*ty][2*tx+1] = x4d(x_b2, x_c2, x_h2 + x_p2, x_w2 + x_q2);
    }
    else{
      MdB[2*ty][2*tx+1] = 0.0;
    }

    if(x_b3 < B && x_c3 < C && (x_h3 + x_p3) < H && (x_w3 + x_q3) < W){
      MdB[2*ty+1][2*tx] = x4d(x_b3, x_c3, x_h3 + x_p3, x_w3 + x_q3);
    }
    else{
      MdB[2*ty+1][2*tx] = 0.0;
    }

    if(x_b4 < B && x_c4 < C && (x_h4 + x_p4) < H && (x_w4 + x_q4) < W){
      MdB[2*ty+1][2*tx+1] = x4d(x_b4, x_c4, x_h4 + x_p4, x_w4 + x_q4);
    }
    else{
      MdB[2*ty+1][2*tx+1] = 0.0;
    }

    __syncthreads();

    for(int k=0; k<TILE_WIDTH_BIG; k++){
        pValue1 += MdA[2*ty][k] * MdB[k][2*tx];
        pValue2 += MdA[2*ty][k] * MdB[k][2*tx+1];
        pValue3 += MdA[2*ty+1][k] * MdB[k][2*tx];
        pValue4 += MdA[2*ty+1][k] * MdB[k][2*tx+1];
    }
    __syncthreads();

  }
  
  if(row1 < numCRows && col1 < numCColumns){
    int y_b1 = bz;
    int y_m1 = row1;
    int y_h1 = col1 / W_out;
    int y_w1 = col1 % W_out;
    if(y_b1 < B && y_m1 < M){
      y4d(y_b1, y_m1, y_h1, y_w1) = pValue1;
    }
  }

  if(row1 < numCRows && col2 < numCColumns){
    int y_b2 = bz;
    int y_m2 = row1;
    int y_h2 = col2 / W_out;
    int y_w2 = col2 % W_out;
    if(y_b2 < B && y_m2 < M){
      y4d(y_b2, y_m2, y_h2, y_w2) = pValue2;
    }
  }

  if(row2 < numCRows && col1 < numCColumns){
    int y_b3 = bz;
    int y_m3 = row2;
    int y_h3 = col1 / W_out;
    int y_w3 = col1 % W_out;
    if(y_b3 < B && y_m3 < M){
      y4d(y_b3, y_m3, y_h3, y_w3) = pValue3;
    }
  }

  if(row2 < numCRows && col2 < numCColumns){
    int y_b4 = bz;
    int y_m4 = row2;
    int y_h4 = col2 / W_out;
    int y_w4 = col2 % W_out;
    if(y_b4 < B && y_m4 < M){
      y4d(y_b4, y_m4, y_h4, y_w4) = pValue4;
    }
  }

  #undef y4d
  #undef x4d
  #undef k4d
}


__global__ void matrixMultiplyShared16(float *k, float *x, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K){

  __shared__ float MdA[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];
  __shared__ float MdB[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, bz = blockIdx.z;

  int row1 = by * TILE_WIDTH_SMALL + 2*ty;
  int row2 = by * TILE_WIDTH_SMALL + 2*ty + 1;
  
  int col1 = bx * TILE_WIDTH_SMALL + 2*tx;
  int col2 = bx * TILE_WIDTH_SMALL + 2*tx + 1;

  int numAColumns = C*K*K;
  int numCRows = M;
  int numCColumns = H_out * W_out;

  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  float pValue1 = 0.0;
  float pValue2 = 0.0;
  float pValue3 = 0.0;
  float pValue4 = 0.0;

  int max_phases = ceil(numAColumns*1.0/TILE_WIDTH_SMALL);
  
  for(int ph=0; ph < max_phases; ph++){  

    int temp_y1 = ph * TILE_WIDTH_SMALL + 2*ty;
    int temp_y2 = ph * TILE_WIDTH_SMALL + 2*ty + 1;

    int temp_x1 = ph * TILE_WIDTH_SMALL + 2*tx;
    int temp_x2 = ph * TILE_WIDTH_SMALL + 2*tx + 1;

    int k_c1 = temp_x1 / (K*K);
    int k_h1 = temp_x1 % (K*K) / K;
    int k_w1 = temp_x1 % (K*K) % K; 
    int k_m1 = row1;

    int k_c2 = temp_x2 / (K*K);
    int k_h2 = temp_x2 % (K*K) / K;
    int k_w2 = temp_x2 % (K*K) % K; 
    int k_m2 = k_m1;

    int k_c3 = temp_x1 / (K*K);
    int k_h3 = temp_x1 % (K*K) / K;
    int k_w3 = temp_x1 % (K*K) % K; 
    int k_m3 = row2;

    int k_c4 = temp_x2 / (K*K);
    int k_h4 = temp_x2 % (K*K) / K;
    int k_w4 = temp_x2 % (K*K) % K; 
    int k_m4 = k_m3;

    if(k_m1 < M && k_c1 < C){
      MdA[2*ty][2*tx] = k4d(k_m1, k_c1, k_h1, k_w1);
    }
    else{
      MdA[2*ty][2*tx] = 0.0;
    }

    if(k_m2 < M && k_c2 < C){
      MdA[2*ty][2*tx+1] = k4d(k_m2, k_c2, k_h2, k_w2);
    }
    else{
      MdA[2*ty][2*tx+1] = 0.0;
    }

   if(k_m3 < M && k_c3 < C){
      MdA[2*ty+1][2*tx] = k4d(k_m3, k_c3, k_h3, k_w3);
    }
    else{
      MdA[2*ty+1][2*tx] = 0.0;
    }

   if(k_m4 < M && k_c4 < C){
      MdA[2*ty+1][2*tx+1] = k4d(k_m4, k_c4, k_h4, k_w4);
    }
    else{
      MdA[2*ty+1][2*tx+1] = 0.0;
    }

    int x_b1 = bz;
    int x_c1 = temp_y1 / (K*K);
    int x_h1 = col1 / W_out;
    int x_w1 = col1 % W_out; 
    int x_p1 = temp_y1 % (K*K) / K;
    int x_q1 = temp_y1 % K;

    int x_b2 = x_b1;
    int x_c2 = x_c1;
    int x_h2 = col2 / W_out;
    int x_w2 = col2 % W_out; 
    int x_p2 = x_p1;
    int x_q2 = x_q1;

    int x_b3 = bz;
    int x_c3 = temp_y2 / (K*K);
    int x_h3 = col1 / W_out;
    int x_w3 = col1 % W_out; 
    int x_p3 = temp_y2 % (K*K) / K;
    int x_q3 = temp_y2 % K;

    int x_b4 = x_b3;
    int x_c4 = x_c3;
    int x_h4 = col2 / W_out;
    int x_w4 = col2 % W_out; 
    int x_p4 = x_p3;
    int x_q4 = x_q3;

    if(x_b1 < B && x_c1 < C && (x_h1 + x_p1) < H && (x_w1 + x_q1) < W){
      MdB[2*ty][2*tx] = x4d(x_b1, x_c1, x_h1 + x_p1, x_w1 + x_q1);
    }
    else{
      MdB[2*ty][2*tx] = 0.0;
    }

    if(x_b2 < B && x_c2 < C && (x_h2 + x_p2) < H && (x_w2 + x_q2) < W){
      MdB[2*ty][2*tx+1] = x4d(x_b2, x_c2, x_h2 + x_p2, x_w2 + x_q2);
    }
    else{
      MdB[2*ty][2*tx+1] = 0.0;
    }

    if(x_b3 < B && x_c3 < C && (x_h3 + x_p3) < H && (x_w3 + x_q3) < W){
      MdB[2*ty+1][2*tx] = x4d(x_b3, x_c3, x_h3 + x_p3, x_w3 + x_q3);
    }
    else{
      MdB[2*ty+1][2*tx] = 0.0;
    }

    if(x_b4 < B && x_c4 < C && (x_h4 + x_p4) < H && (x_w4 + x_q4) < W){
      MdB[2*ty+1][2*tx+1] = x4d(x_b4, x_c4, x_h4 + x_p4, x_w4 + x_q4);
    }
    else{
      MdB[2*ty+1][2*tx+1] = 0.0;
    }

    __syncthreads();

    for(int k=0; k<TILE_WIDTH_SMALL; k++){
        pValue1 += MdA[2*ty][k] * MdB[k][2*tx];
        pValue2 += MdA[2*ty][k] * MdB[k][2*tx+1];
        pValue3 += MdA[2*ty+1][k] * MdB[k][2*tx];
        pValue4 += MdA[2*ty+1][k] * MdB[k][2*tx+1];
    }
    __syncthreads();

  }
  
  if(row1 < numCRows && col1 < numCColumns){
    int y_b1 = bz;
    int y_m1 = row1;
    int y_h1 = col1 / W_out;
    int y_w1 = col1 % W_out;
    if(y_b1 < B && y_m1 < M){
      y4d(y_b1, y_m1, y_h1, y_w1) = pValue1;
    }
  }

  if(row1 < numCRows && col2 < numCColumns){
    int y_b2 = bz;
    int y_m2 = row1;
    int y_h2 = col2 / W_out;
    int y_w2 = col2 % W_out;
    if(y_b2 < B && y_m2 < M){
      y4d(y_b2, y_m2, y_h2, y_w2) = pValue2;
    }
  }

  if(row2 < numCRows && col1 < numCColumns){
    int y_b3 = bz;
    int y_m3 = row2;
    int y_h3 = col1 / W_out;
    int y_w3 = col1 % W_out;
    if(y_b3 < B && y_m3 < M){
      y4d(y_b3, y_m3, y_h3, y_w3) = pValue3;
    }
  }

  if(row2 < numCRows && col2 < numCColumns){
    int y_b4 = bz;
    int y_m4 = row2;
    int y_h4 = col2 / W_out;
    int y_w4 = col2 % W_out;
    if(y_b4 < B && y_m4 < M){
      y4d(y_b4, y_m4, y_h4, y_w4) = pValue4;
    }
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

    if(C == 1){
        dim3 blockDim(TILE_WIDTH_SMALL/2, TILE_WIDTH_SMALL/2, 1);
        dim3 gridDim(ceil(out_size*1.0/TILE_WIDTH_SMALL), ceil(M*1.0/TILE_WIDTH_SMALL), B);
        matrixMultiplyShared16<<<gridDim, blockDim>>>(w.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K);
    }
    else{
        dim3 blockDim(TILE_WIDTH_BIG/2, TILE_WIDTH_BIG/2, 1);
        dim3 gridDim(ceil(out_size*1.0/TILE_WIDTH_BIG), ceil(M*1.0/TILE_WIDTH_BIG), B);
        matrixMultiplyShared32<<<gridDim, blockDim>>>(w.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K);
    }

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