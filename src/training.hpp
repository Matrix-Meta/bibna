// src/training.hpp
#pragma once
#include "common.hpp"

class TrainingTools {
  sycl::queue& q;

 public:
  TrainingTools(sycl::queue& queue) : q(queue) {}

  // Cross Entropy Loss & Gradient
  // Loss = -log(softmax(logits)[target])
  // Grad = softmax(logits) - one_hot(target)
  float cross_entropy(Tensor& logits, int target_idx, Tensor& d_logits) {
    // Copy Logits to Host for Softmax calculation
    std::vector<float> h_logits(logits.size);
    q.memcpy(h_logits.data(), logits.data, logits.size * sizeof(float)).wait();

    float max_val = -1e9;
    for (float v : h_logits) max_val = std::max(max_val, v);

    float sum_exp = 0.0f;
    std::vector<float> probs(logits.size);
    for (size_t i = 0; i < h_logits.size(); ++i) {
      probs[i] = std::exp(h_logits[i] - max_val);
      sum_exp += probs[i];
    }

    float loss = 0.0f;
    // 計算 Gradient 並回填 d_logits
    std::vector<float> h_grad(logits.size);
    for (size_t i = 0; i < probs.size(); ++i) {
      probs[i] /= sum_exp;  // Normalize

      float target_val = (i == target_idx) ? 1.0f : 0.0f;
      h_grad[i] = (probs[i] - target_val);  // dL/dz = p - y

      if (i == target_idx) {
        loss = -std::log(probs[i] + 1e-9f);
      }
    }

    // Copy Grad back to Device
    q.memcpy(d_logits.data, h_grad.data(), logits.size * sizeof(float)).wait();
    return loss;
  }

  // SGD Optimizer: W = W - lr * grad
  void sgd_step(Tensor& weights, float lr) {
    q.parallel_for(sycl::range<1>(weights.size), [=, w_ptr = weights.data,
                                                  g_ptr = weights.grad](
                                                     sycl::id<1> idx) {
       w_ptr[idx] -= lr * g_ptr[idx];
       g_ptr[idx] = 0.0f;  // 重置梯度，非常重要！
     }).wait();
  }
};
