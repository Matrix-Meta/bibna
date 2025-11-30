// src/spiking_unit.hpp
#pragma once
#include "common.hpp"
#include "kernels.hpp"

class SpikingUnit
{
    sycl::queue &q;

  public:
    float theta;
    float alpha;
    
    SpikingUnit(sycl::queue &queue, float th = 1.0f, float a = 2.0f) : q(queue), theta(th), alpha(a)
    {
    }

    Tensor forward(Tensor &v)
    {
        Tensor s(v.size, q);
        kernels::spike_fwd_bwd(q, v.data, s.data, nullptr, nullptr, v.size, theta, alpha, false);
        return s;
    }

    // 無狀態反向傳播：需要輸入 v
    void backward(Tensor &grad_s, Tensor &v_in)
    {
        // 計算 Surrogate Gradient 並回傳給 v_in.grad
        if (v_in.grad) {
            kernels::spike_fwd_bwd(q, v_in.data, nullptr, grad_s.data, v_in.grad, v_in.size, theta, alpha, true);
        }
    }
};
