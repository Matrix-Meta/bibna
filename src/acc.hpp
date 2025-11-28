#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "common.hpp"

class ConflictMonitor {
  sycl::queue& q;

 public:
  ConflictMonitor(sycl::queue& queue) : q(queue) {}

  // 計算 Logits 的 Entropy (熵)
  // Entropy = - sum(p * log(p))
  // 高 Entropy 代表模型很困惑 (分佈很平)；低 Entropy 代表模型很確定 (尖峰分佈)
  float calculate_uncertainty(Tensor& logits) {
    // 1. Copy Logits to Host (為了計算方便，實際高效能應寫 Kernel)
    std::vector<float> h_logits(logits.size);
    q.memcpy(h_logits.data(), logits.data, logits.size * sizeof(float)).wait();

    // 2. Softmax (與 Decoding 重複，但在 ACC 中我們需要精確的機率分佈)
    float max_val = -1e9;
    for (float v : h_logits) max_val = std::max(max_val, v);

    std::vector<float> probs(logits.size);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < h_logits.size(); ++i) {
      probs[i] = std::exp(h_logits[i] - max_val);
      sum_exp += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
      probs[i] /= sum_exp;
    }

    // 3. Calculate Entropy
    float entropy = 0.0f;
    for (float p : probs) {
      if (p > 1e-9f) {  // 避免 log(0)
        entropy -= p * std::log(p);
      }
    }

    return entropy;
  }

  // 簡單的衝突判斷：如果 Entropy 超過閾值，視為高衝突
  bool is_high_conflict(float entropy, float threshold = 2.0f) {
    return entropy > threshold;
  }
};
