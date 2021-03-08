//////////////////////////////////////////////////////////////////////////////
/*
INRF2d_gpu.cu.cc

Code written by Ryan Cecil under Stacey Levine, Ph.D.
Duquesne University 2020

Purpose: Cuda Implementation of the INRF operator for Tensorflow

Implementation based on INRF equation found in the following paper:
Evidence for the intrinsically nonlinear
nature of receptive fields in vision by Marcelo Bertalmio,
Alex Gomez-Villa, Adrian Martin, Javier Vazquez-Corral, David Kane, & Jesus
Malo. Link: https://www.nature.com/articles/s41598-020-73113-0

Current Nonlinearity used is ReLU
*//////////////////////////////////////////////////////////////////////////////

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/user_ops/INRF2d_gpu0.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cuda_profiler_api.h"
#include <stdio.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

//Function to divide then round up
inline int divUp(int length, int block_size)
{
  return (length + block_size - 1) / block_size;
}

//INRF2d
/////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void INRF2dKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor3<T>::ConstTensor m,
    const typename Tensor3<T>::ConstTensor w,
    const typename Tensor3<T>::ConstTensor g,
    const typename Tensor2<T>::ConstTensor lambda,
    typename Tensor2<T>::Tensor out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w,
    const unsigned int shared_x_size)
{
  //Compute length of padding
  const unsigned int pad = num_w-1;

  //Get lengths of filters and channels
  const unsigned int num_channels = x.dimensions()[1];
  const unsigned int filter_length = w.dimensions()[0];
  const unsigned int shared_filter_length = filter_length*num_channels;

  //Get feature to compute
  const unsigned int idf = blockIdx.y;

  //Get thread id and location in out image
  const unsigned int tid = threadIdx.x;
  const unsigned int start_idx = blockDim.x*blockIdx.x;
  const unsigned int idx = threadIdx.x + start_idx;

  //Find location within batch and compute shared memory indice
  const unsigned int row_num = idx/out_col_num;
  const unsigned int image_num = idx/out_image_size;
  const unsigned int start_row_num = start_idx/out_col_num;
  const unsigned int start_image_num = start_idx/out_image_size;
  const unsigned int tid_index = tid + (row_num - start_row_num) * pad + (image_num - start_image_num)*pad*col_num;
  const unsigned int x_start = start_idx + (start_idx/out_col_num) * pad + (start_idx/out_image_size)*pad*col_num;

  //Declare shared memory
  extern __shared__ T m_filter[];
  T *w_filter = m_filter + shared_filter_length;
  T *g_filter = w_filter + shared_filter_length;
  T *lambda_shared = g_filter+shared_filter_length;
  T *x_shared = lambda_shared + num_channels;

  //Load lambda
  if(tid < num_channels)
  {
    lambda_shared[tid] = lambda(tid,idf);
  }

  //Load filters
  if(tid < shared_filter_length)
  {
    m_filter[tid] = m(tid%filter_length,tid/filter_length,idf);
    w_filter[tid] = w(tid%filter_length,tid/filter_length,idf);
    g_filter[tid] = g(tid%filter_length,tid/filter_length,idf);
  }

  //Apply filters to channels
  T output = 0.0;
  for(int idc = 0; idc < x.dimensions()[1]; idc++)
  {
    //Load single channel of x into shared memory
    unsigned int xtid = tid;
    while(xtid < shared_x_size && x_start+xtid < x.dimensions()[0])
    {
      x_shared[xtid] = x(x_start+xtid,idc);
      xtid += blockDim.x;
    }
    //Make sure x is entirely loaded
    __syncthreads();

    //Check that we are within image bounds
    if(idx < out.dimensions()[0])
    {

      T wi = 0.0;
      T mi = 0.0;
      T gi = 0.0;
      //Apply m and g filters
      for(int idkx = 0; idkx < num_w; idkx++)
      {
        for(int idky = 0; idky < num_w; idky++)
        {
          mi += m_filter[idc*filter_length + idkx*num_w + idky]*x_shared[tid_index+idkx*col_num+idky];
          gi += g_filter[idc*filter_length + idkx*num_w + idky]*x_shared[tid_index+idkx*col_num+idky];
        }
      }
      //Compute shift value, apply relu, then apply w weights
      for(int idwx = 0; idwx < num_w; idwx++)
      {
        for(int idwy = 0; idwy < num_w; idwy++)
        {
          //T shift = gi - x_shared[tid_index+idwx*col_num+idwy] - gi;
          T shift = gi - x_shared[tid_index+idwx*col_num+idwy];
          if(shift > 0) wi += w_filter[idc*filter_length + idwx*num_w + idwy]*shift;
        }
      }
      //output += mi - lambda_shared[idc]*wi;
      output += mi + lambda_shared[idc]*wi;
    }
    //Store output
    out(idx,idf) = output;
    //Make sure all threads are finished before we load in the next channel
    __syncthreads();
  }

};

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void INRF2dFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &m,
    const typename Tensor3<T>::ConstTensor &w,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &lambda,
    typename Tensor2<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
    {
    // For proifiling the kernel during testing
    cudaProfilerStart();

    //Assign number of threads to each block
    unsigned int threads_per_block = 512;
    unsigned int block_count = divUp(out.dimensions()[0], threads_per_block);

    //block x dimension corresponds to location in batch
    //block y dimension corresponds to feature
    dim3 numBlocks(block_count,out.dimensions()[1]);

    //Compute amount of shared memory for x
    unsigned int num_images = divUp(threads_per_block,out_image_size);
    unsigned int num_rows = divUp(threads_per_block,out_col_num) + (num_images+1)*(num_w-1);
    unsigned int shared_x_size = num_rows*col_num;
    if(shared_x_size > x.dimensions()[0]) shared_x_size = x.dimensions()[0];

    //Shared memory for filters, lambda, and x
    unsigned int smem_size = (3*w.dimensions()[0]*w.dimensions()[1] + lambda.dimensions()[0] + shared_x_size)*sizeof(T);

    //Launch Cuda Kernel
    INRF2dKernel<T>
        <<<numBlocks, threads_per_block, smem_size, d.stream()>>>(x,m,w,g,lambda,out, out_col_num, out_image_size, col_num, num_w, shared_x_size);
    cudaProfilerStop();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct INRF2dFunctor<GPUDevice, float>;
//template struct INRF2dFunctor<GPUDevice, int32>;



//INRF2dGradW
///////////////////////////////////////////////////////////////////////////////

//Quick kernel to set grad of filter to zero
template<typename T>
__global__ void SetFilterOutToZero(typename Tensor3<T>::Tensor out)
{

  const unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

  if(idx < out.dimensions()[0])
  {
    for(int idf = 0; idf < out.dimensions()[2]; idf++)
    {
      for(int idc = 0; idc < out.dimensions()[1]; idc++)
      {
        out(idx,idc,idf) = 0;
      }
    }
  }
};

//Reduce data using warp
template <typename T, unsigned int blockSize>
__device__ void
warpReduce(volatile T *sdata, unsigned int tid)
{
  if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
  if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
  if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
  if(blockSize >=   8) sdata[tid] += sdata[tid +  4];
  if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
  if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
}

//Reduce data in shared memory
template <typename T, unsigned int blockSize>
__device__ void reduce(volatile T *sdata,unsigned int tid)
{
  if(blockSize == 1024){
    if (tid < 512)
    {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
      }
      __syncthreads();
    }
    if (blockSize >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
      }
      __syncthreads();
    }
    if (blockSize >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
      }
      __syncthreads();
    }
    if (tid < 32) warpReduce<T,blockSize>(sdata, tid);
  }

template<typename T, unsigned int blockSize>
__global__ void INRF2dGradWKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor3<T>::ConstTensor g,
    const typename Tensor2<T>::ConstTensor lambda,
    const typename Tensor2<T>::ConstTensor grad,
    typename Tensor3<T>::Tensor out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
{

  //Compute length of padding
  const unsigned int pad = num_w-1;

  //Get lengths of filters and channels
  const unsigned int num_channels = x.dimensions()[1];
  const unsigned int filter_length = g.dimensions()[0];
  const unsigned int shared_filter_length = filter_length*num_channels;

  //Get feature to compute
  const unsigned int idf = blockIdx.y;

  //Get thread id and location in out image
  const unsigned int tid = threadIdx.x;
  const unsigned int start_idx = blockDim.x*blockIdx.x;
  const unsigned int idx = threadIdx.x + start_idx;

  //Location in x
  const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;

  //Declare shared memory
  extern __shared__ T g_filter[];
  T *lambda_shared = g_filter+shared_filter_length;
  T *w_out_shared = lambda_shared + num_channels;
  T *grad_shared = w_out_shared + blockDim.x;

  //Set shared memory for output to zero
  w_out_shared[tid] = 0.0;

  //Load Lambda
  if(tid < num_channels)
  {
    lambda_shared[tid] = lambda(tid,idf);
  }

  //Load g filter
  if(tid < shared_filter_length)
  {
    g_filter[tid] = g(tid%filter_length,tid/filter_length,idf);
  }

  //Load gradient
  if(idx < grad.dimensions()[0])
  {
    grad_shared[tid] = grad(idx,idf);
  }

  //Make sure everything is loaded into shared memory
  __syncthreads();

  //Compute grad w for each channel
  for (int idc = 0; idc < num_channels; idc++)
  {
    //Check that we are still within the images
    if(idx < grad.dimensions()[0])
    {
      // Apply g filter
      T gi = 0.0;
      for(int idkx = 0; idkx < num_w; idkx++)
      {
        for(int idky = 0; idky < num_w; idky++)
        {
          gi += g_filter[idkx*num_w + idky]*x(index+idkx*col_num+idky,idc);
        }
      }
      //Compute shift, apply relu, then add gradient
      for(int idwx = 0; idwx < num_w; idwx++)
      {
        for(int idwy = 0; idwy <num_w; idwy++)
        {
          //T shift = x(index+idwx*col_num+idwy,idc) - gi;
          T shift = gi - x(index+idwx*col_num+idwy,idc);
          if(shift > 0) w_out_shared[tid] = shift*grad_shared[tid];
          else w_out_shared[tid] = 0.0;
          //Wait for all gradient values to be loaded into shared memory
          __syncthreads();
          //Sum gradient values in shared memory
          reduce<T,blockSize>(w_out_shared, tid);

          //Add summed gradients to output and multiply by lamdba
          //if(tid == 0) CudaAtomicAdd(&out(idwx*num_w + idwy,idc,idf),-lambda_shared[idc]*w_out_shared[tid]);
          if(tid == 0) CudaAtomicAdd(&out(idwx*num_w + idwy,idc,idf),lambda_shared[idc]*w_out_shared[tid]);
        }
      }
    }
  }
};



// GPU functor for grad w
template <typename T>
void INRF2dGradWFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &lambda,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor3<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
    {
  //For kernel profiling
  cudaProfilerStart();

  //Assign threads per block
  const unsigned int threads_per_block = 512;
  const unsigned int num_zero_blocks = divUp(out.dimensions()[0], threads_per_block);

  //Set output memory to 0 values
  SetFilterOutToZero<T>
      <<<num_zero_blocks, threads_per_block, 0, d.stream()>>>(out);

  unsigned int block_count = divUp(grad.dimensions()[0], threads_per_block);
  dim3 numBlocks(block_count,grad.dimensions()[1]);

  //Declared shared memory for g filter, lambda, grad, and output data
  unsigned int smem_size = (g.dimensions()[0]*g.dimensions()[1] + lambda.dimensions()[0] + 2*threads_per_block)*sizeof(T);

  //Launch Kernel
  INRF2dGradWKernel<T,threads_per_block>
      <<<numBlocks, threads_per_block, smem_size, d.stream()>>>(x,g,lambda,grad,out,out_col_num,out_image_size,col_num,num_w);
  cudaProfilerStop();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct INRF2dGradWFunctor<GPUDevice, float>;
//template struct INRF2dGradWFunctor<GPUDevice, int32>;


//INRF2d GradG
//////////////////////////////////////////////////////////////////////////////
template <typename T,unsigned int blockSize>
__global__ void INRF2dGradGKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor3<T>::ConstTensor w,
    const typename Tensor3<T>::ConstTensor g,
    const typename Tensor2<T>::ConstTensor lambda,
    const typename Tensor2<T>::ConstTensor grad,
    typename Tensor3<T>::Tensor out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w,
    const unsigned int shared_x_size)
{
  //Compute length of padding
  const unsigned int pad = num_w-1;

  //Get lengths of filters and channels
  const unsigned int num_channels = x.dimensions()[1];
  const unsigned int filter_length = w.dimensions()[0];
  const unsigned int shared_filter_length = filter_length*num_channels;

  //Get feature to compute
  const unsigned int idf = blockIdx.y;

  //Get thread id and location in out image
  const unsigned int tid = threadIdx.x;
  const unsigned int start_idx = blockDim.x*blockIdx.x;
  const unsigned int idx = threadIdx.x + start_idx;

  //Find location within batch and compute shared memory indice
  const unsigned int row_num = idx/out_col_num;
  const unsigned int image_num = idx/out_image_size;
  const unsigned int start_row_num = start_idx/out_col_num;
  const unsigned int start_image_num = start_idx/out_image_size;
  const unsigned int tid_index = tid + (row_num - start_row_num) * pad + (image_num - start_image_num)*pad*col_num;
  const unsigned int x_start = start_idx + (start_idx/out_col_num) * pad + (start_idx/out_image_size)*pad*col_num;

  //Declare shared memory
  extern __shared__ T w_filter[];
  T *g_filter = w_filter+shared_filter_length;
  T *lambda_shared = g_filter+shared_filter_length;
  T *grad_shared = lambda_shared + num_channels;
  T *x_shared = grad_shared + blockDim.x;
  T *sdata = x_shared + shared_x_size;

  //Load Lambda
  if(tid < num_channels)
  {
    lambda_shared[tid] = lambda(tid,idf);
  }

  //Load filters
  if(tid < shared_filter_length)
  {
    w_filter[tid] = w(tid%filter_length,tid/filter_length,idf);
    g_filter[tid] = g(tid%filter_length,tid/filter_length,idf);
  }

  //Load grad
  if(idx < grad.dimensions()[0])
  {
    grad_shared[tid] = grad(idx,idf);
  }

  //Compute grad values with respect to g for each channel
  for (int idc = 0; idc < out.dimensions()[1]; idc++)
  {
    //Load idc channel of x into shared memory
    unsigned int xtid = tid;
    while(xtid < shared_x_size && x_start+xtid < x.dimensions()[0])
    {
      x_shared[xtid] = x(x_start+xtid,idc);
      xtid += blockDim.x;
    }

    //Set shared data locations to zero
    if(tid < filter_length)
    {
      sdata[tid] = 0.0;
    }

    //Make sure everything is loaded
    __syncthreads();

    //Check that we are within image bounds
    if(idx < grad.dimensions()[0])
    {
      //Apply g filter
      T gi = 0.0;
      for(int idkx = 0; idkx < num_w; idkx++)
      {
        for(int idky = 0; idky < num_w; idky++)
        {
          gi += g_filter[idc*filter_length + idkx*num_w + idky]*x_shared[tid_index+idkx*col_num+idky];
        }
      }
      //Compute shift, apply relu, if positive compute g gradient value
      for(int idwx = 0; idwx < num_w; idwx++)
      {
        for(int idwy = 0; idwy < num_w; idwy++)
        {
          //T shift = x_shared[tid_index+idwx*col_num+idwy] - gi;
          T shift = gi - x_shared[tid_index+idwx*col_num+idwy];
          if(shift > 0)
          {
            for(int idkx = 0; idkx < num_w; idkx++)
            {
              for(int idky = 0; idky < num_w; idky++)
              {
                //Compute g gradient
                T output = w_filter[idc*filter_length + idwx*num_w + idwy]*x_shared[tid_index+idkx*col_num+idky]*grad_shared[tid];
                // Add to shared data
                CudaAtomicAdd(&sdata[idkx*num_w + idky],output);
              }
            }
          }
        }
      }
    }
    //Make sure all values have been added to sdata
    __syncthreads();
    //Add values to gradient output
    if(tid < filter_length)
    {
      CudaAtomicAdd(&out(tid,idc,idf),lambda_shared[idc]*sdata[tid]);
    }
  }
};


// GPU functor for grad G
template <typename T>
void INRF2dGradGFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &w,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &lambda,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor3<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
    {
      //For kernel profiling
      cudaProfilerStart();

      //Assign threads
      const unsigned int threads_per_block = 512;
      const unsigned int num_zero_blocks = divUp(out.dimensions()[0], threads_per_block);

      //Set output memory to zero
      SetFilterOutToZero<T>
          <<<num_zero_blocks, threads_per_block, 0, d.stream()>>>(out);

      unsigned int block_count = divUp(grad.dimensions()[0], threads_per_block);
      dim3 numBlocks(block_count,grad.dimensions()[1]);

      //Compute size of shared x memory
      unsigned int num_images = divUp(threads_per_block,out_image_size);
      unsigned int num_rows = divUp(threads_per_block,out_col_num) + (num_images+1)*(num_w-1);
      unsigned int shared_x_size = num_rows*col_num;

      if(shared_x_size > x.dimensions()[0]) shared_x_size = x.dimensions()[0];

      //Declare shared memory for filters, lambda, grad, x, and sdata
      unsigned int smem_size = (2*w.dimensions()[0]*w.dimensions()[1] + lambda.dimensions()[0] + threads_per_block + shared_x_size + w.dimensions()[0])*sizeof(T);

      //Launch kernel
      INRF2dGradGKernel<T,threads_per_block>
          <<<numBlocks, threads_per_block, smem_size, d.stream()>>>(x,w,g,lambda,grad,out,out_col_num,out_image_size,col_num,num_w,shared_x_size);
      cudaProfilerStop();
  }

// Explicitly instantiate functors for the types of OpKernels registered.
template struct INRF2dGradGFunctor<GPUDevice, float>;
//template struct INRF2dGradGFunctor<GPUDevice, int32>;

//INRF2dGradX
///////////////////////////////////////////////////////////////////////////////

//Quick kernel to set x grad to zero
template<typename T>
__global__ void ClearXOutput(typename Tensor2<T>::Tensor out)
{

  const unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

  for(int idc = 0; idc < out.dimensions()[1]; idc++)
  {
    if(idx < out.dimensions()[0])
    {
      out(idx,idc) = 0.0;
    }
  }
};



// INRF2d Gradient with respect to input
template<typename T>
__global__ void INRF2dGradXKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor3<T>::ConstTensor w,
    const typename Tensor3<T>::ConstTensor g,
    const typename Tensor2<T>::ConstTensor lambda,
    const typename Tensor2<T>::ConstTensor grad,
    typename Tensor2<T>::Tensor out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w,
    const unsigned int shared_x_size)
{
  //Compute length of padding
  const unsigned int pad = num_w-1;

  //Get lengths of filters and channels
  const unsigned int num_channels = x.dimensions()[1];
  const unsigned int filter_length = w.dimensions()[0];
  const unsigned int shared_filter_length = filter_length*num_channels;

  //Get feature to compute
  const unsigned int idf = blockIdx.y;

  //Get thread id and location in out image
  const unsigned int tid = threadIdx.x;
  const unsigned int start_idx = blockDim.x*blockIdx.x;
  const unsigned int idx = threadIdx.x + start_idx;

  //Find location within batch and compute shared memory indice
  const unsigned int row_num = idx/out_col_num;
  const unsigned int image_num = idx/out_image_size;
  const unsigned int start_row_num = start_idx/out_col_num;
  const unsigned int start_image_num = start_idx/out_image_size;
  const unsigned int tid_index = tid + (row_num - start_row_num) * pad + (image_num - start_image_num)*pad*col_num;
  const unsigned int x_start = start_idx + (start_idx/out_col_num) * pad + (start_idx/out_image_size)*pad*col_num;

  //Declare shared memory
  extern __shared__ T w_filter[];
  T *g_filter = w_filter+shared_filter_length;
  T *lambda_shared = g_filter+shared_filter_length;
  T *grad_shared = lambda_shared + num_channels;
  T *x_shared = grad_shared + blockDim.x;
  T *x_shared_out = x_shared + shared_x_size;

  //Load lambda
  if(tid < num_channels)
  {
    lambda_shared[tid] = lambda(tid,idf);
  }

  //Store filter and lambda weights into shared memory
  if(tid < shared_filter_length)
  {
    w_filter[tid] = w(tid%filter_length,tid/filter_length,idf);
    g_filter[tid] = g(tid%filter_length,tid/filter_length,idf);
  }

  //Load grad
  if(idx < grad.dimensions()[0])
  {
    grad_shared[tid] = grad(idx,idf);
  }

  //Compute x grad for each channel
  for(int idc = 0; idc < x.dimensions()[1]; idc++)
  {
    //Load x channel into shared memory
    unsigned int xtid = tid;
    while(xtid < shared_x_size && x_start+xtid < x.dimensions()[0])
    {
      x_shared[xtid] = x(x_start+xtid,idc);
      x_shared_out[xtid] = 0.0;
      xtid += blockDim.x;
    }

    //Make sure everything is loaded
    __syncthreads();

    //Ensure we are still within image bounds
    if(idx < grad.dimensions()[0])
    {
      //Apply g filter
      T gi = 0.0;
      for(int idkx = 0; idkx < num_w; idkx++)
      {
        for(int idky = 0; idky < num_w; idky++)
        {
          gi += g_filter[idc*filter_length + idkx*num_w + idky]*x_shared[tid_index+idkx*col_num+idky];
        }
      }
      //Compute shift, apply relu, if positive then compute x grad
      for(int idwx = 0; idwx < num_w; idwx++)
       {
         for(int idwy = 0; idwy < num_w; idwy++)
         {
           //T shift = x_shared[tid_index+idwx*col_num+idwy] - gi;
           T shift = gi - x_shared[tid_index+idwx*col_num+idwy];
           if (shift > 0)
           {
             for(int idkx = 0; idkx < num_w; idkx++)
             {
               for(int idky = 0; idky < num_w; idky++)
               {
                 if(idkx == idwx && idky == idwy){
                   //T output = -w_filter[idc*filter_length + idwx*num_w + idwy]*(1-g_filter[idc*filter_length + idkx*num_w + idky])*grad_shared[tid];
                   T output = w_filter[idc*filter_length + idwx*num_w + idwy]*(g_filter[idc*filter_length + idkx*num_w + idky]-1)*grad_shared[tid];
                   //Add grad to shared memory
                   CudaAtomicAdd(&x_shared_out[tid_index+idkx*col_num+idky],output);
                 }
                 else
                 {
                   T output = w_filter[idc*filter_length + idwx*num_w + idwy]*g_filter[idc*filter_length + idkx*num_w + idky]*grad_shared[tid];
                   //Add grad to shared memory
                   CudaAtomicAdd(&x_shared_out[tid_index+idkx*col_num+idky],output);
                 }
               }
             }
           }
         }
       }
     }
     //Make sure all gradient values are added
     __syncthreads();
     //Add gradient values to output gradient
     xtid = tid;
     while(xtid < shared_x_size && x_start+xtid < x.dimensions()[0])
     {
       CudaAtomicAdd(&out(x_start+xtid,idc),lambda_shared[idc]*x_shared_out[xtid]);
       xtid += blockDim.x;
     }
   }
 };


// GPU functor for grad input
template <typename T>
void INRF2dGradXFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &m,
    const typename Tensor3<T>::ConstTensor &w,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &lambda,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor2<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
    {
      //For kernel profiling
      cudaProfilerStart();

      //Assign threads
      unsigned int threads_per_block = 512;
      unsigned int block_count_zero = divUp(out.dimensions()[0], threads_per_block);

      //Clear output memory
      ClearXOutput<T>
           <<<block_count_zero, threads_per_block, 0, d.stream()>>>(out);

      unsigned int block_count = divUp(grad.dimensions()[0], threads_per_block);
      dim3 numBlocks(block_count,grad.dimensions()[1]);

      //Compute size of shared memory for x
      unsigned int num_images = divUp(threads_per_block,out_image_size);
      unsigned int num_rows = divUp(threads_per_block,out_col_num) + (num_images+1)*(num_w-1);
      unsigned int shared_x_size = num_rows*col_num;
      if(shared_x_size > x.dimensions()[0]) shared_x_size = x.dimensions()[0];

      //Declare shared memory for filters, lambda, grad, x, and shared output data
      unsigned int smem_size = (2*w.dimensions()[0]*w.dimensions()[1] + lambda.dimensions()[0] + threads_per_block + 2*shared_x_size)*sizeof(T);

      //Launch kernel
      INRF2dGradXKernel<T>
          <<<numBlocks, threads_per_block, smem_size, d.stream()>>>(x,w,g,lambda,grad,out,out_col_num,out_image_size,col_num,num_w,shared_x_size);

      cudaProfilerStop();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct INRF2dGradXFunctor<GPUDevice, float>;
//template struct INRF2dGradXFunctor<GPUDevice, int32>;


//INRF2d GradL
/////////////////////////////////////////////////////////////////////////////

//Quick kernel to set output lamba gradient to zero
template<typename T>
__global__ void SetLambdaOutToZero(typename Tensor2<T>::Tensor out)
{

  const unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

  if(idx < out.dimensions()[1])
  {
    for(int idc = 0; idc < out.dimensions()[0]; idc++)
    {
      out(idc,idx) = 0.0;
    }
  }
};

template<typename T, unsigned int blockSize>
__global__ void INRF2dGradLKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor3<T>::ConstTensor w,
    const typename Tensor3<T>::ConstTensor g,
    const typename Tensor2<T>::ConstTensor grad,
    typename Tensor2<T>::Tensor out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
{

  //Compute length of padding
  const unsigned int pad = num_w-1;

  //Get lengths of filters and channels
  const unsigned int num_channels = x.dimensions()[1];
  const unsigned int filter_length = w.dimensions()[0];
  const unsigned int shared_filter_length = filter_length*num_channels;

  //Get feature to compute
  const unsigned int idf = blockIdx.y;

  //Get thread id and location in image
  const unsigned int tid = threadIdx.x;
  const unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;

  //Declare shared memory
  extern __shared__ T w_filter[];
  T *g_filter = w_filter+shared_filter_length;
  T *grad_shared = g_filter + shared_filter_length;
  T *sdata = grad_shared + blockDim.x;

  //Set shared data to zero
  sdata[tid] = 0.0;

  //Load filters
  if(tid < shared_filter_length)
  {
    w_filter[tid] = w(tid%filter_length,tid/filter_length,idf);
    g_filter[tid] = g(tid%filter_length,tid/filter_length,idf);
  }

  //Load grad
  if(idx < grad.dimensions()[0])
  {
    grad_shared[tid] = grad(idx,idf);
  }

  //Make sure everything is loaded into shared memory
  __syncthreads();

  //Compute gradients for each channel
  for(int idc = 0; idc < out.dimensions()[0]; idc++)
  {
    //Check that we are within image bounds
    if(idx < grad.dimensions()[0])
    {
      //Apply g filter
      T gi = 0.0;
      for(int idkx = 0; idkx < num_w; idkx++)
      {
        for(int idky = 0; idky < num_w; idky++)
        {
          gi += g_filter[idc*filter_length + idkx*num_w + idky]*x(index+idkx*col_num+idky,idc);
        }
      }
      //Compute shift, apply relu, if positive compute lambda gradient
      T output = 0.0;
      for(int idwx = 0; idwx < num_w; idwx++)
      {
        for(int idwy = 0; idwy < num_w; idwy++)
        {
          //T shift = x(index+idwx*col_num+idwy,idc) - gi;
          //T shift = x(index+idwx*col_num+idwy,idc) - gi;
          T shift = gi - x(index+idwx*col_num+idwy,idc);
          if(shift > 0)
          {
            output += w_filter[idc*filter_length + idwx*num_w + idwy]*shift;
          }
        }
      }
      //Store gradient value in shared memory
      sdata[tid] = output*grad_shared[tid];
      //Make sure all values are stored
      __syncthreads();
      //Sum values in shared memory
      reduce<T,blockSize>(sdata, tid);
      //Add summed values to output
      ///if(tid == 0) CudaAtomicAdd(&out(idc,idf),-sdata[tid]);
      if(tid == 0) CudaAtomicAdd(&out(idc,idf),sdata[tid]);
    }
  }
};

// GPU functor for grad L
template <typename T>
void INRF2dGradLFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &w,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor2<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
    {
      //For kernel profiling
      cudaProfilerStart();

      //Assign threads
      const unsigned int threads_per_block = 512;
      const unsigned int num_zero_blocks = divUp(out.dimensions()[1], threads_per_block);

      //Set output memory to zero
      SetLambdaOutToZero<T>
          <<<num_zero_blocks, threads_per_block, 0, d.stream()>>>(out);

      unsigned int block_count = divUp(grad.dimensions()[0], threads_per_block);
      dim3 numBlocks(block_count,grad.dimensions()[1]);

      //Declare shared memory for filters, grad, and shared data
      unsigned int smem_size = (2*w.dimensions()[0]*w.dimensions()[1] + 2*threads_per_block)*sizeof(T);

      //Launch Kernel
      INRF2dGradLKernel<T,threads_per_block>
          <<<numBlocks, threads_per_block, smem_size, d.stream()>>>(x,w,g,grad,out,out_col_num,out_image_size,col_num,num_w);
      cudaProfilerStop();
  }

// Explicitly instantiate functors for the types of OpKernels registered.
template struct INRF2dGradLFunctor<GPUDevice, float>;
//template struct INRF2dGradLFunctor<GPUDevice, int32>;


#endif GOOGLE_CUDA
