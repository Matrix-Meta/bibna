// src/bitlinear.hpp
#pragma once
#include "common.hpp"
#include "kernels.hpp"

class BitLinear {
  int in_dim, out_dim;
  sycl::queue& q;

 public:
  Tensor weights;    // Float weights (master)
  Tensor weights_q;  // Quantized weights
  float* scale;      // Quantization scale

  // Cache for backward
  Tensor* input_cache = nullptr;
  float* rstd_cache = nullptr;

  BitLinear(int in, int out, sycl::queue& queue)
      : in_dim(in),
        out_dim(out),
        q(queue),
        weights(in * out, queue),
        weights_q(in * out, queue) {
    weights.random_init();
    scale = sycl::malloc_shared<float>(1, q);
    rstd_cache = sycl::malloc_shared<float>(1024, q);  // 假設最大 batch 1024
  }

  // Move Constructor
  BitLinear(BitLinear&& other) noexcept
      : in_dim(other.in_dim),
        out_dim(other.out_dim),
        q(other.q),
        weights(std::move(other.weights)),
        weights_q(std::move(other.weights_q)),
        scale(other.scale),
        rstd_cache(other.rstd_cache),
        input_cache(other.input_cache) {
    other.scale = nullptr;
    other.rstd_cache = nullptr;
    other.input_cache = nullptr;
  }

  // Delete Copy and Move Assignment
  BitLinear(const BitLinear&) = delete;
  BitLinear& operator=(const BitLinear&) = delete;
  BitLinear& operator=(BitLinear&&) = delete;

  ~BitLinear() {
    if (scale) sycl::free(scale, q);
    if (rstd_cache) sycl::free(rstd_cache, q);
  }

  Tensor forward(Tensor& input) {
    int batch = input.size / in_dim;
    input_cache = &input;  // Keep reference

    // 1. RMSNorm
    Tensor x_norm(input.size, q);
    kernels::rms_norm_fwd(q, input.data, x_norm.data, rstd_cache, batch,
                          in_dim);

    // 2. Quantize Weights
    kernels::quantize_weights(q, weights.data, weights_q.data, scale,
                              weights.size);

    // 3. Matrix Multiplication (Naive for demo)
    Tensor output(batch * out_dim, q);
    float current_scale = *scale;

    // Capture by value
    int d_in = in_dim;
    int d_out = out_dim;

    q.parallel_for(sycl::range<2>(batch, out_dim), [=, w_ptr = weights_q.data,
                                                    x_ptr = x_norm.data,
                                                    out_ptr = output.data](
                                                       sycl::id<2> idx) {
       int b = idx[0];
       int o = idx[1];
       float sum = 0.0f;
       for (int i = 0; i < d_in; ++i) {
         // x @ W_q^T
         sum += x_ptr[b * d_in + i] * w_ptr[o * d_in + i];
       }
       // De-quantize
       out_ptr[b * d_out + o] = sum * current_scale;
     }).wait();

    return output;
  }

  // Real STE Backward (Straight-Through Estimator)
  void backward(Tensor& grad_output) {
    int batch = grad_output.size / out_dim;
    float current_scale = *scale;

    // Capture dimensions by value for kernels to avoid 'this' capture
    int d_in = in_dim;
    int d_out = out_dim;

    // 1. Calculate Input Gradients (dL/dx = dL/dy * W)
    // 為了簡單，我們假設 input_cache->grad 已經被歸零或初始化
    if (input_cache && input_cache->grad) {
      q.parallel_for(sycl::range<2>(batch, d_in),
                     [=, grad_y = grad_output.data, w = weights.data,
                      grad_x = input_cache->grad](sycl::id<2> idx) {
                       int b = idx[0];
                       int i = idx[1];
                       float sum = 0.0f;
                       for (int o = 0; o < d_out; ++o) {
                         // Backprop through linear: sum(grad_y * weight)
                         // 注意：這裡使用 float weight 近似 (STE)
                         sum += grad_y[b * d_out + o] * w[o * d_in + i];
                       }
                       // 這裡略過了 RMSNorm 的 backward (假設它是
                       // identity)，在初期實驗是可以接受的
                       grad_x[b * d_in + i] += sum;
                     })
          .wait();
    }

    // 2. Calculate Weight Gradients (dL/dW = x^T * dL/dy)
    // 這是讓模型學習的關鍵！
    // 我們使用 Atomic Add 或者簡單的迴圈 (為了教學清晰，這裡用 Naive
    // 寫法，效能非最佳但邏輯正確)
    q.parallel_for(sycl::range<2>(d_out, d_in), [=,
                                                     grad_y = grad_output.data,
                                                     x = input_cache->data,
                                                     w_grad = weights.grad](
                                                        sycl::id<2> idx) {
       int o = idx[0];
       int i = idx[1];
       float sum = 0.0f;
       for (int b = 0; b < batch; ++b) {
         // dL/dw_oi = sum_over_batch( x_bi * dL/dy_bo )
         // 記得乘上量化的 scale (STE 修正)
         sum += x[b * d_in + i] * grad_y[b * d_out + o];
       }
       // 累積梯度 (Accumulate)
       w_grad[o * d_in + i] += sum * current_scale;
     }).wait();
  }
};
