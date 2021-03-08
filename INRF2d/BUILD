# Description:
# Tells Bazel what files to build for operator.

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "INRF2d_gpu.so",
    srcs = ["INRF2d_gpu.cc",
            "INRF2d_gpu.h",
            ],
    gpu_srcs = ["INRF2d_gpu.cu.cc",
		"INRF2d_gpu.h",
                ],
)
