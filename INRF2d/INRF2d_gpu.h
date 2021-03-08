//////////////////////////////////////////////////////////////////////////////
/*
INRF2d_gpu.h

Code written by Ryan Cecil under Stacey Levine, Ph.D.
Duquesne University 2020

Purpose: Header file for implementation of the INRF operator for Tensorflow

Implementation based on INRF equation found in the following paper:
Evidence for the intrinsically nonlinear
nature of receptive fields in vision by Marcelo Bertalmio,
Alex Gomez-Villa, Adrian Martin, Javier Vazquez-Corral, David Kane, & Jesus
Malo. Link: https://www.nature.com/articles/s41598-020-73113-0

Current Nonlinearity used is ReLU
*//////////////////////////////////////////////////////////////////////////////
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


template<typename T>
using Tensor3 = tensorflow::TTypes<T,3>;
template<typename T>
using Tensor2 = tensorflow::TTypes<T,2>;


//Note that the gradient with respect to the m filter can be computed using
//tensorflow convolution operations


// Computes INRF2d
template<typename Device, typename T>
struct INRF2dFunctor {
  void operator()(const Device& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &m,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to w weights
template<typename Device, typename T>
struct INRF2dGradWFunctor {
  void operator()(const Device& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  typename Tensor3<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to input
template<typename Device, typename T>
struct INRF2dGradXFunctor {
  void operator()(const Device& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &m,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to g filter
template<typename Device, typename T>
struct INRF2dGradGFunctor {
  void operator()(const Device& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor3<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

//Computes INRF gradient with respect to lambda weights
template<typename Device, typename T>
struct INRF2dGradLFunctor {
  void operator()(const Device& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

//Specialize functors for GpuDevice.
#if GOOGLE_CUDA

// Compute INRF on GPU
template<typename T>
struct INRF2dFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &m,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to w weights on GPU
template<typename T>
struct INRF2dGradWFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor3<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to input on GPU
template<typename T>
struct INRF2dGradXFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &m,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to g weights on GPU
template<typename T>
struct INRF2dGradGFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &lambda,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor3<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

// Computes INRF gradient with respect to lambda weights on GPU
template<typename T>
struct INRF2dGradLFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor3<T>::ConstTensor &w,
                  const typename Tensor3<T>::ConstTensor &g,
                  const typename Tensor2<T>::ConstTensor &grad,
                  typename Tensor2<T>::Tensor &out,
                  const unsigned int out_col_num,
                  const unsigned int out_image_size,
                  const unsigned int col_num,
                  const unsigned int num_w);
};

#endif GOOGLE_CUDA
