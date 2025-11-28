// src/spiking_unit.hpp
#pragma once
#include "common.hpp"
#include "kernels.hpp"

class SpikingUnit {
  sycl::queue& q;
  float theta;
  float alpha;
  Tensor* v_cache = nullptr;

 public:
  SpikingUnit(sycl::queue& queue, float th = 1.0f, float a = 2.0f)
      : q(queue), theta(th), alpha(a) {}

  Tensor forward(Tensor& v) {
    v_cache = &v;
    Tensor s(v.size, q);
    kernels::spike_fwd_bwd(q, v.data, s.data, nullptr, nullptr, v.size, theta,
                           alpha, false);
    return s;
  }

  void backward(Tensor& grad_s) {
    if (!v_cache) return;
    // 計算 Surrogate Gradient 並回傳給 v.grad
    kernels::spike_fwd_bwd(q, v_cache->data, nullptr, grad_s.data,
                           v_cache->grad, v_cache->size, theta, alpha, true);
  }
};
