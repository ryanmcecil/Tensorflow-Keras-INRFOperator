//////////////////////////////////////////////////////////////////////////////
/*
INRF2d_gpu.cc

Code written by Ryan Cecil under Stacey Levine, Ph.D.
Duquesne University 2020

Purpose: Registers operators and
and contains C++ implementation of the INRF operator for Tensorflow

Note that the implementation is for an INRF with valid padding so the inputted
tensor must be padded before calling the operation

Implementation based on INRF equation found in the following paper:
Evidence for the intrinsically nonlinear
nature of receptive fields in vision by Marcelo Bertalmio,
Alex Gomez-Villa, Adrian Martin, Javier Vazquez-Corral, David Kane, & Jesus
Malo. Link: https://www.nature.com/articles/s41598-020-73113-0

Current Nonlinearity used is ReLU
*//////////////////////////////////////////////////////////////////////////////

#define EIGEN_USE_THREADS
#include "tensorflow/core/user_ops/INRF2d_gpu0.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <stdio.h>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


//Register INRF and INRF gradient Operators
/////////////////////////////////////////////////////////////////////////////
constexpr auto INRFCommonIOs = R"doc(
x: Input to the INRF. Must be a 4D tensor.
m: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
w: Convolution weights. Must be a 4D tensor. Fourth dimension defines the number of
filters while the third defines the number of channels.
g: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
lamda: Two dimensional tensor. First dimensions defines number of channels
while last defines number of features.
output: Output of the INRF.
)doc";

constexpr auto INRFGradWCommonIOs = R"doc(
x: Input to the INRF. Must be a 4D tensor.
g: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
lamda: First dimensions defines number of channels while last defines number of features
grad: Backpropagated gradient.
output: Gradient that should be backpropagated to the filter weights.
)doc";

constexpr auto INRFGradXCommonIOs = R"doc(
x: Input to the INRF. Must be a 4D tensor.
m: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
w: Convolution weights. Must be a 4D tensor. Fourth dimension defines the number of
    filters while the third defines the number of channels.
g: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
lamda: Two dimensional tensor. First dimensions defines number of channels
while last defines number of features.
grad: Backpropagated gradient.
output: Gradient that should be backpropagated to the input.
)doc";

constexpr auto INRFGradGCommonIOs = R"doc(
x: Input to the INRF. Must be a 4D tensor.
w: Convolution weights. Must be a 4D tensor. Fourth dimension defines the number of
    filters while the third defines the number of channels.
g: Convolution input to the INRF. Must be a 4D tensor. The last
    dimension defines the number of features.
lamda: Two dimensional tensor. First dimensions defines number of channels
while last defines number of features.
grad: Backpropagated gradient.
output: Gradient that should be backpropagated to g convolution.
)doc";

constexpr auto INRFGradLCommonIOs = R"doc(
x: Input to the INRF. Must be a 4D tensor.
w: Convolution weights. Must be a 4D tensor. Fourth dimension defines the number of
    filters while the third defines the number of channels.
g: Convolution input to the INRF. Must be a 4D tensor. Fourth dimension defines the number of
    filters while the third defines the number of channels.
grad: Two dimensional tensor. First dimensions defines number of channels
while last defines number of features.
output: Gradient that should be backpropagated to the lambda weights.
)doc";

constexpr auto INRFCommonAttrs = R"doc(
)doc";

//Register operators, set inputs/ouputs, and set output shapes
REGISTER_OP("INRF2d")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("m: T")
    .Input("w: T")
    .Input("g: T")
    .Input("lamda: T")
    .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(c->input(0),0),c->UnknownDim(),c->UnknownDim(),c->Dim(c->input(1),3)});TF_MUST_USE_RESULT;
    c->set_output(0,output_shape);
  return Status::OK();})
  .Doc(strings::StrCat(R"doc(
Computes an INRF)doc",
INRFCommonIOs,
INRFCommonAttrs));

REGISTER_OP("INRF2dGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("g: T")
    .Input("lamda: T")
    .Input("grad: T")
    .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(c->input(1),0),c->Dim(c->input(1),1),c->Dim(c->input(1),2),c->Dim(c->input(1),3)});TF_MUST_USE_RESULT;
    c->set_output(0,output_shape);
  return Status::OK();})
  .Doc(strings::StrCat(R"doc(
Computes an INRF)doc",
INRFGradWCommonIOs,
INRFCommonAttrs));

REGISTER_OP("INRF2dGradX")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("m: T")
    .Input("w: T")
    .Input("g: T")
    .Input("lamda: T")
    .Input("grad: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      c->set_output(0, c->input(0));
      return Status::OK();})
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the INRF
output to the corresponding input x.
)doc",
INRFGradXCommonIOs,
INRFCommonAttrs));

REGISTER_OP("INRF2dGradG")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Input("g: T")
    .Input("lamda: T")
    .Input("grad: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      c->set_output(0, c->input(2));
      return Status::OK();})
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the INRF
output to the corresponding input g.
)doc",
INRFGradGCommonIOs,
INRFCommonAttrs));

REGISTER_OP("INRF2dGradL")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Input("g: T")
    .Input("grad: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(c->input(0),3),c->Dim(c->input(3),3)});TF_MUST_USE_RESULT;
      c->set_output(0,output_shape);
    return Status::OK();})
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the INRF
output to the lambda weights.
)doc",
INRFGradLCommonIOs,
INRFCommonAttrs));

// INRF2d
///////////////////////////////////////////////////////////

//Implements cpu version of INRF operator
template <typename T>
struct INRF2dFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    const typename Tensor2<T>::ConstTensor &x,
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
    //Get size of padding and number of channels
    const unsigned int pad = num_w-1;
    const unsigned int num_channels = x.dimensions()[1];

    //For each feature, compute INRF at each pixel across all channels
    for(int idf = 0; idf < out.dimensions()[1]; idf++)
    {
      for(int idx = 0; idx < out.dimensions()[0]; idx++)
      {
        //Find location in input x
        const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;
        T output = 0.0;
        for(int idc = 0; idc < x.dimensions()[1]; idc++)
        {
          T wi = 0.0;
          T mi = 0.0;
          T gi = 0.0;

          //Apply m and g filters
          for(int idkx = 0; idkx < num_w; idkx++)
          {
            for(int idky = 0; idky < num_w; idky++)
            {
              mi += m(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
              gi += g(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
            }
          }

          //Compute shift, apply ReLU, then apply w weights
          for(int idwx = 0; idwx < num_w; idwx++)
          {
            for(int idwy = 0; idwy < num_w; idwy++)
            {
              //T shift = x(index+idwx*col_num+idwy,idc)-gi;
              T shift = gi - x(index+idwx*col_num+idwy,idc);
              if(shift > 0) wi += w(idwx*num_w + idwy,idc,idf)*shift;
            }
          }
          //output += mi - lambda(idc,idf)*wi;
          output += mi + lambda(idc,idf)*wi;
        }
        out(idx,idf) = output;
      }
    }
  }
};

// OpKernel definition for INRF2d
template <typename Device, typename T>
class INRF2dOp : public OpKernel {
 public:
  explicit INRF2dOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensors
    const Tensor& x = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& w = context->input(2);
    const Tensor& g = context->input(3);
    const Tensor& lambda = context->input(4);


    //Check for correct sizes of inputs
    OP_REQUIRES(context, lambda.dim_size(1) == w.dim_size(3),
              errors::Unimplemented("Lambda must have the same number of features as w. \nLambda: ",lambda.dim_size(1), "\nw: ",w.dim_size(3)));

    OP_REQUIRES(context, m.dim_size(3) == w.dim_size(3) && m.dim_size(3) == g.dim_size(3),
              errors::Unimplemented("All filters must have the same number of features. \nm: ",m.dim_size(1),"w: ",w.dim_size(3),"g: ",g.dim_size(3)));

    OP_REQUIRES(context, m.dim_size(2) == w.dim_size(2) && m.dim_size(2) == g.dim_size(2) && m.dim_size(2) == x.dim_size(3),
              errors::Unimplemented("All filters must have the same number of channels as the input. \nm: ",m.dim_size(2),"w: ",w.dim_size(2),"g: ",g.dim_size(2), "x: ",x.dim_size(3)));

    OP_REQUIRES(context, m.dim_size(0) == w.dim_size(0)  && m.dim_size(0) == g.dim_size(0),
                        errors::Unimplemented("m, w, and g filters must have the same shape. \nm: ",m.dim_size(0), "\nw: ",w.dim_size(0),"\ng: ",g.dim_size(0)));


    // Create the output tensor
    TensorShape output_shape({x.dim_size(0), x.dim_size(1)-(w.dim_size(0)-1), x.dim_size(2)-(w.dim_size(0)-1), w.dim_size(3)});
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    //Flatten output to 2d
    auto out_tensor_map = output_tensor->flat_inner_dims<T,2>();

    //Get image sizes
    const unsigned int out_col_num = output_shape.dim_size(2);
    const unsigned int out_image_size = output_shape.dim_size(1)*output_shape.dim_size(2);
    const unsigned int col_num = x.dim_size(2);
    const unsigned int num_w = w.dim_size(0);

    //Flatten tensos and compute INRF
    INRF2dFunctor<Device, T>()(
        context->eigen_device<Device>(),
        x.flat_inner_dims<T,2>(),
        m.flat_inner_dims<T,3>(),
        w.flat_inner_dims<T,3>(),
        g.flat_inner_dims<T,3>(),
        lambda.tensor<T,2>(),
        out_tensor_map,
        out_col_num,
        out_image_size,
        col_num,
        num_w);
      }
};

//Register CPU functor
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2d").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      INRF2dOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

//Register GPU functor
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template class INRF2dFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2d").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      INRF2dOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif



//INRF with Respect to W
//////////////////////////////////////////////////////////////

// Computes INRF gradient with respect to w weights on CPU
template <typename T>
struct INRF2dGradWFunctor<CPUDevice, T>
{
  void operator()(const CPUDevice& d,
    const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &lambda,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor3<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
  {
    //Get size of padding and channels
    const unsigned int pad = num_w-1;
    const unsigned int num_channels = x.dimensions()[1];

    //Zero out output memory
    for(int idf = 0; idf < grad.dimensions()[1]; idf++)
    {
      for (int idc = 0; idc < num_channels; idc++)
      {
        for(int idwx = 0; idwx < num_w; idwx++)
        {
          for(int idwy = 0; idwy <num_w; idwy++)
          {
            out(idwx*num_w + idwy,idc,idf) = 0.0;
          }
        }
      }
    }

    //For each feature, for each channel, for each pixel in the gradient with
    //respect to the output, compute the gradient with respect to each w filter
    //weight
    for(int idf = 0; idf < grad.dimensions()[1]; idf++)
    {
      for (int idc = 0; idc < num_channels; idc++)
      {
        for(int idx = 0; idx < grad.dimensions()[0]; idx++)
        {
          //Get location in input image
          const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;
          //Apply g filter to input
          T gi = 0.0;
          for(int idkx = 0; idkx < num_w; idkx++)
          {
            for(int idky = 0; idky < num_w; idky++)
            {
              gi += g(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
            }
          }
          //Compute shift, apply nonlinearity, apply w filter weights
          for(int idwx = 0; idwx < num_w; idwx++)
          {
            for(int idwy = 0; idwy <num_w; idwy++)
            {
              //T shift = x(index+idwx*col_num+idwy,idc) - gi;
              T shift = gi - x(index+idwx*col_num+idwy,idc);
              if(shift > 0) out(idwx*num_w + idwy,idc,idf) += shift*grad(idx,idf);
            }
          }
        }
        //Multiply sum of w gradients by lambda value
        for(int idwx = 0; idwx < num_w; idwx++)
        {
          for(int idwy = 0; idwy <num_w; idwy++)
          {
            //out(idwx*num_w + idwy,idc,idf) = -lambda(idc,idf)*out(idwx*num_w + idwy,idc,idf);
            out(idwx*num_w + idwy,idc,idf) = lambda(idc,idf)*out(idwx*num_w + idwy,idc,idf);
          }
        }
      }
    }
  }
};

// OpKernel definition for INRF2d gradient with respect to w weights
template <typename Device, typename T>
class INRF2dGradWOp : public OpKernel {
 public:
  explicit INRF2dGradWOp(OpKernelConstruction* context) : OpKernel(context){}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensors
    const Tensor& x = context->input(0);
    const Tensor& g = context->input(1);
    const Tensor& lambda = context->input(2);
    const Tensor& grad_out = context->input(3);


    Tensor *output_w = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, g.shape(),
                                                     &output_w));
    //Get image sizes
    const unsigned int out_col_num = grad_out.dim_size(2);
    const unsigned int out_image_size = grad_out.dim_size(1)*grad_out.dim_size(2);
    const unsigned int col_num = x.dim_size(2);
    const unsigned int num_w = g.dim_size(0);

    //Flatten and call the computation
    auto grad_w_tensor_map = output_w->flat_inner_dims<T,3>();

    INRF2dGradWFunctor<Device, T>()(
        context->eigen_device<Device>(),
        x.flat_inner_dims<T,2>(),
        g.flat_inner_dims<T,3>(),
        lambda.tensor<T,2>(),
        grad_out.flat_inner_dims<T,2>(),
        grad_w_tensor_map,
        out_col_num,
        out_image_size,
        col_num,
        num_w);
      }
};

//Register CPU kernel
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradW").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      INRF2dGradWOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

//Register GPU kernel
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template class INRF2dGradWFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradW").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      INRF2dGradWOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA


//INRF with respect to X
////////////////////////////////////////////////////////////////////

//Compute INRF gradient with respect to input on CPU
template <typename T>
struct INRF2dGradXFunctor<CPUDevice, T>
{
  void operator()(const CPUDevice& d,
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
    const unsigned int num_w)
  {
    //Get size of padding and channels
    const unsigned int pad = num_w-1;
    const unsigned int num_channels = x.dimensions()[1];

    //Zero out output memory
    for(int idc = 0; idc < out.dimensions()[1]; idc++)
    {
      for(int idx = 0; idx < out.dimensions()[0]; idx++)
      {
        out(idx,idc) = 0.0;
      }
    }

    //For each channel, for each feature, for each pixel in the output gradient,
    //compute INRF with respect to input values
    for(int idc = 0; idc < x.dimensions()[1]; idc++)
    {
      for(int idf = 0; idf < grad.dimensions()[1]; idf++)
      {
        for(int idx = 0; idx < grad.dimensions()[0]; idx++)
        {
          //Get location in input x
          const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;

          //Apply g filter
          T gi = 0.0;
          for(int idkx = 0; idkx < num_w; idkx++)
          {
            for(int idky = 0; idky < num_w; idky++)
            {
              gi += g(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
            }
          }

          //Compute shift, apply nonlinearity
          for(int idwx = 0; idwx < num_w; idwx++)
           {
             for(int idwy = 0; idwy < num_w; idwy++)
             {
               //T shift = x(index+idwx*col_num+idwy,idc) - gi;
               T shift = gi - x(index+idwx*col_num+idwy,idc);
               //printf("%d\n",shift);
               if (shift > 0)
               {
                 //For each g filter weight, compute gradient with respect to x
                 for(int idkx = 0; idkx < num_w; idkx++)
                 {
                   for(int idky = 0; idky < num_w; idky++)
                   {
                     if(idkx == idwx && idky == idwy){
                       //out(index+idkx*col_num+idky,idc) -= lambda(idc,idf)*w(idwx*num_w + idwy,idc,idf)*(1-g(idkx*num_w + idky,idc,idf))*grad(idx,idf);
                       out(index+idkx*col_num+idky,idc) += lambda(idc,idf)*w(idwx*num_w + idwy,idc,idf)*(g(idkx*num_w + idky,idc,idf)-1)*grad(idx,idf);
                     }
                     else
                     {
                       //out(index+idkx*col_num+idky,idc) += lambda(idc,idf)*w(idwx*num_w + idwy,idc,idf)*g(idkx*num_w + idky,idc,idf)*grad(idx,idf);
                       out(index+idkx*col_num+idky,idc) += lambda(idc,idf)*w(idwx*num_w + idwy,idc,idf)*g(idkx*num_w + idky,idc,idf)*grad(idx,idf);
                     }
                   }
                 }
               }
             }
           }
         }
       }
     }
   }
 };


// OpKernel definition for INRF2d gradient with respect to input
template <typename Device, typename T>
class INRF2dGradXOp : public OpKernel
{
 public:
  explicit INRF2dGradXOp(OpKernelConstruction* context) : OpKernel(context){}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& x = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& w = context->input(2);
    const Tensor& g = context->input(3);
    const Tensor& lambda = context->input(4);
    const Tensor& grad_out = context->input(5);

    Tensor *output_x = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(),
                                                     &output_x));

    // Get image sizes
    const unsigned int out_col_num = grad_out.dim_size(2);
    const unsigned int out_image_size = grad_out.dim_size(1)*grad_out.dim_size(2);
    const unsigned int col_num = x.dim_size(2);
    const unsigned int num_w = w.dim_size(0);

    // Do the computation and flatten tensor dimensions
    auto grad_x_tensor_map = output_x->flat_inner_dims<T,2>();

    INRF2dGradXFunctor<Device, T>()(
        context->eigen_device<Device>(),
        x.flat_inner_dims<T,2>(),
        m.flat_inner_dims<T,3>(),
        w.flat_inner_dims<T,3>(),
        g.flat_inner_dims<T,3>(),
        lambda.tensor<T,2>(),
        grad_out.flat_inner_dims<T,2>(),
        grad_x_tensor_map,
        out_col_num,
        out_image_size,
        col_num,
        num_w);
      }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradX").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      INRF2dGradXOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template class INRF2dGradXFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradX").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      INRF2dGradXOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

//INRF with respect to G
//////////////////////////////////////////////////////////////////////

//Compute INRF gradient with respect to g weights on CPU
template <typename T>
struct INRF2dGradGFunctor<CPUDevice, T>
{
  void operator()(const CPUDevice& d,
    const typename Tensor2<T>::ConstTensor &x,
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

    //Get size of padding
    const unsigned int pad = num_w-1;

    //Zero out output memory
    for(int idf = 0; idf < out.dimensions()[2]; idf++)
    {
      for (int idc = 0; idc < out.dimensions()[1]; idc++)
      {
        for (int idx = 0; idx < out.dimensions()[0]; idx++)
        {
          out(idx,idc,idf) = 0.0;
        }
      }
    }

    //For each feature, for each channel, for each pixel in gradient with
    //respect to INRF ouput, compute grad with respect to g filter weights
    for(int idf = 0; idf < out.dimensions()[2]; idf++)
    {
      for (int idc = 0; idc < out.dimensions()[1]; idc++)
      {
        for(int idx = 0; idx < grad.dimensions()[0]; idx++)
        {
          // Get location in input x
          const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;

          //Apply g filter
          T gi = 0.0;
          for(int idkx = 0; idkx < num_w; idkx++)
          {
            for(int idky = 0; idky < num_w; idky++)
            {
              gi += g(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
            }
          }

          //Compute shift, apply nonlinearity
          for(int idwx = 0; idwx < num_w; idwx++)
          {
            for(int idwy = 0; idwy < num_w; idwy++)
            {
              //T shift = x(index+idwx*col_num+idwy,idc) - gi;
              T shift = gi - x(index+idwx*col_num+idwy,idc);
              if(shift > 0)
              {
                //Compute gradient with respect to each g filter weight
                for(int idkx = 0; idkx < num_w; idkx++)
                {
                  for(int idky = 0; idky < num_w; idky++)
                  {
                    out(idkx*num_w + idky,idc,idf)+=w(idwx*num_w + idwy,idc,idf)*x(index+idkx*col_num+idky,idc)*grad(idx,idf);
                  }
                }
              }
            }
          }
        }
        //Multiply gradients by lambda weight
        for(int idkx = 0; idkx < num_w; idkx++)
        {
          for(int idky = 0; idky < num_w; idky++)
          {
            out(idkx*num_w + idky,idc,idf) *= lambda(idc,idf);
          }
        }
      }
    }
  }
};


// OpKernel definition for INRF gradient with respect to g weights
template <typename Device, typename T>
class INRF2dGradGOp : public OpKernel {
 public:
  explicit INRF2dGradGOp(OpKernelConstruction* context) : OpKernel(context){}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensors
    const Tensor& x = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& g = context->input(2);
    const Tensor& lambda = context->input(3);
    const Tensor& grad_out = context->input(4);

    Tensor *output_g = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, g.shape(),
                                                     &output_g));

    // Get image sizes
    const unsigned int out_col_num = grad_out.dim_size(2);
    const unsigned int out_image_size = grad_out.dim_size(1)*grad_out.dim_size(2);
    const unsigned int col_num = x.dim_size(2);
    const unsigned int num_w = w.dim_size(0);

    // Flatten dimensions and the computation
    auto grad_g_tensor_map = output_g->flat_inner_dims<T,3>();

    INRF2dGradGFunctor<Device, T>()(
        context->eigen_device<Device>(),
        x.flat_inner_dims<T,2>(),
        w.flat_inner_dims<T,3>(),
        g.flat_inner_dims<T,3>(),
        lambda.tensor<T,2>(),
        grad_out.flat_inner_dims<T,2>(),
        grad_g_tensor_map,
        out_col_num,
        out_image_size,
        col_num,
        num_w);
      }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradG").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      INRF2dGradGOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template class INRF2dGradGFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradG").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      INRF2dGradGOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

// INRF with respect to lambda
//////////////////////////////////////////////////////////////////////////////

//Compute INRF with respect to lambda on the CPU
template <typename T>
struct INRF2dGradLFunctor<CPUDevice, T>
{
  void operator()(const CPUDevice& d,
    const typename Tensor2<T>::ConstTensor &x,
    const typename Tensor3<T>::ConstTensor &w,
    const typename Tensor3<T>::ConstTensor &g,
    const typename Tensor2<T>::ConstTensor &grad,
    typename Tensor2<T>::Tensor &out,
    const unsigned int out_col_num,
    const unsigned int out_image_size,
    const unsigned int col_num,
    const unsigned int num_w)
  {

    //Get padding size
    const unsigned int pad = num_w-1;

    //Set output memory to zero
    for(int idc = 0; idc < out.dimensions()[0]; idc++)
    {
      for(int idf = 0; idf < out.dimensions()[1]; idf++)
      {
        out(idc,idf) = 0.0;
      }
    }

    //For each feature, for each channel, for each pixel in gradient
    //with respect to output of INRF, compute gradient with respect to lambda
    for(int idf = 0; idf < out.dimensions()[1]; idf++)
    {
      for(int idc = 0; idc < out.dimensions()[0]; idc++)
      {
        for(int idx = 0; idx < grad.dimensions()[0]; idx++)
        {
          //Get location in input x
          const unsigned int index = idx + (idx/out_col_num) * pad + (idx/out_image_size)*pad*col_num;
          T output = 0.0;
          T gi = 0.0;
          //Apply g filter to input
          for(int idkx = 0; idkx < num_w; idkx++)
          {
            for(int idky = 0; idky < num_w; idky++)
            {
              gi += g(idkx*num_w + idky,idc,idf)*x(index+idkx*col_num+idky,idc);
            }
          }

          //Compute shift, apply nonlinearity, compute lamdba gradient
          for(int idwx = 0; idwx < num_w; idwx++)
          {
            for(int idwy = 0; idwy < num_w; idwy++)
            {
              //T shift = x(index+idwx*col_num+idwy,idc) - gi;
              T shift = gi - x(index+idwx*col_num+idwy,idc);
              if(shift > 0)
              {
                output += w(idwx*num_w + idwy,idc,idf)*shift;
              }
            }
          }
          //out(idc,idf) -= output*grad(idx,idf);
          out(idc,idf) += output*grad(idx,idf);
        }
      }
    }
  }
};

// OpKernel definition for INRF gradient with respect to lambda weights
template <typename Device, typename T>
class INRF2dGradLOp : public OpKernel
{
 public:
  explicit INRF2dGradLOp(OpKernelConstruction* context) : OpKernel(context){}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensors
    const Tensor& x = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& g = context->input(2);
    const Tensor& grad_out = context->input(3);

    TensorShape output_shape({x.dim_size(3), grad_out.dim_size(3)});

    Tensor *output_lambda = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_lambda));

    //Get image sizes
    const unsigned int out_col_num = grad_out.dim_size(2);
    const unsigned int out_image_size = grad_out.dim_size(1)*grad_out.dim_size(2);
    const unsigned int col_num = x.dim_size(2);
    const unsigned int num_w = w.dim_size(0);

    //Flatten tensors and do the computation
    auto grad_lambda_tensor_map = output_lambda->tensor<T,2>();

    INRF2dGradLFunctor<Device, T>()(
        context->eigen_device<Device>(),
        x.flat_inner_dims<T,2>(),
        w.flat_inner_dims<T,3>(),
        g.flat_inner_dims<T,3>(),
        grad_out.flat_inner_dims<T,2>(),
        grad_lambda_tensor_map,
        out_col_num,
        out_image_size,
        col_num,
        num_w);
      }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradL").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      INRF2dGradLOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template class INRF2dGradGFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("INRF2dGradL").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      INRF2dGradLOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
