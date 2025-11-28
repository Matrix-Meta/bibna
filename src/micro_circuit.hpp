// src/micro_circuit.hpp
#pragma once
#include <utility>

#include "bitlinear.hpp"
#include "spiking_unit.hpp"

class MicroCircuit {
  sycl::queue& q;
  int input_dim;
  int hidden_dim;  // MicroCircuit 內部的神經元數量

  // 組件
  BitLinear w_in;
  BitLinear w_rec;
  BitLinear w_out;
  SpikingUnit spikes;

  // 狀態暫存 (State Cache)
  // 為了簡單起見，這裡只演示單步推理或 BPTT 的最後一步
  // 實際訓練需要儲存所有時間步的狀態以進行反向傳播
  Tensor* s_prev = nullptr;

 public:
  MicroCircuit(int dim, int hidden, sycl::queue& queue)
      : q(queue),
        input_dim(dim),
        hidden_dim(hidden),
        w_in(dim, hidden, queue),
        w_rec(hidden, hidden, queue),
        w_out(hidden, dim, queue),  // 輸出維度設為與輸入相同，以便做殘差相加
        spikes(queue, 1.0f, 2.0f)   // Theta=1.0, Alpha=2.0
  {
    // 初始化 Recurrent Weights 為較小的值，避免初始爆炸
    // 這裡可以手動再除以一個係數
  }

  // 單步前向傳播 (One Step Forward)
  // x_t: 當前輸入 [batch, dim]
  // s_prev_t: 上一步的脈衝狀態 [batch, hidden]
  std::pair<Tensor, Tensor> forward_step(Tensor& x_t, Tensor& s_prev_t) {
    // 1. 計算電流
    Tensor u_in = w_in.forward(x_t);
    Tensor u_rec = w_rec.forward(s_prev_t);

    // 2. 融合
    Tensor v_t(u_in.size, q);
    float* u_in_ptr = u_in.data;
    float* u_rec_ptr = u_rec.data;
    float* v_ptr = v_t.data;

    q.parallel_for(sycl::range<1>(v_t.size), [=](sycl::id<1> idx) {
       v_ptr[idx] = u_in_ptr[idx] + u_rec_ptr[idx];
     }).wait();

    // 3. 激發脈衝 (這就是新的 State)
    Tensor s_t = spikes.forward(v_t);

    // 4. 輸出投影與殘差
    Tensor proj = w_out.forward(s_t);
    Tensor output(x_t.size, q);

    float* out_ptr = output.data;
    float* x_ptr = x_t.data;
    float* proj_ptr = proj.data;

    q.parallel_for(sycl::range<1>(output.size), [=](sycl::id<1> idx) {
       out_ptr[idx] = x_ptr[idx] + proj_ptr[idx];
     }).wait();

    // 這裡會觸發 Tensor 的 Move Constructor，效率很高
    return {std::move(output), std::move(s_t)};
  }
};
