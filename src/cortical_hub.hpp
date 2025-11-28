// src/cortical_hub.hpp
#pragma once
#include "bitlinear.hpp"
#include "common.hpp"
#include "kernels.hpp"

class CorticalHub {
  sycl::queue& q;
  int input_dim;     // 來自 MicroCircuit 的隱藏層維度
  int memory_dim;    // Fast Weights 的維度 (通常等於 input_dim 或更小)
  int num_circuits;  // 連接了幾個 MicroCircuit

  // 投影層
  BitLinear w_k;
  BitLinear w_v;
  BitLinear w_gate;

  // 參數
  float lambda = 0.95f;  // 衰減
  float eta = 0.1f;      // 學習率
  float clip = 5.0f;     // 防爆炸鉗制

 public:
  Tensor M_state;  // Fast Weights [Batch, MemDim, MemDim]

  CorticalHub(int in_dim, int mem_dim, int n_circuits, sycl::queue& queue)
      : q(queue),
        input_dim(in_dim),
        memory_dim(mem_dim),
        num_circuits(n_circuits),
        w_k(in_dim, mem_dim, queue),
        w_v(in_dim, mem_dim, queue),
        w_gate(in_dim, 1, queue),  // Gate 輸出純量
        M_state(0, queue)          // 暫時初始化為 0，稍後在 init_memory 設定
  {
    // w_gate 的輸出需要經過 sigmoid，我們在 forward 處理
  }

  void init_memory(int batch_size) {
    // 重置/初始化 Fast Weights
    // 注意：這裡使用 move assignment 重新分配 Tensor
    Tensor new_m(batch_size * memory_dim * memory_dim, q);
    q.fill(new_m.data, 0.0f, new_m.size).wait();
    M_state = std::move(new_m);
  }

  // 更新 Fast Weights
  // inputs: [Batch * NumCircuits, InputDim] (所有 MicroCircuit 的輸出展平)
  void update_memory(Tensor& inputs) {
    int total_tokens = inputs.size / input_dim;
    int batch = total_tokens / num_circuits;

    // 1. 投影 Key, Value, Gate
    Tensor k = w_k.forward(inputs);            // [B*N, MemDim]
    Tensor v = w_v.forward(inputs);            // [B*N, MemDim]
    Tensor g_logits = w_gate.forward(inputs);  // [B*N, 1]

    // 2. 計算 Gate Sigmoid
    Tensor g(g_logits.size, q);
    kernels::sigmoid_fwd(q, g_logits.data, g.data, g.size);

    // 3. 執行 Fast Weights Update Kernel
    // M_state 在此地被就地更新 (In-place update)
    kernels::fast_weight_update(q, M_state.data, k.data, v.data, g.data, batch,
                                num_circuits, memory_dim, lambda, eta, clip);
  }

  // 簡單的讀取測試 (Read Probe)
  // 為了驗證記憶是否有效，我們計算 Query * M * Key^T 的一部分，或者簡單的 M *
  // Query 這裡實作簡單的 y = M * q_vec (假設 q_vec 是全 1 向量來測試能量)
  // 僅用於除錯/監控
  float get_memory_norm() {
    // 計算 M 的 L2 Norm (Host 端)
    std::vector<float> h_M(M_state.size);
    q.memcpy(h_M.data(), M_state.data, M_state.size * sizeof(float)).wait();

    float sum_sq = 0.0f;
    for (float val : h_M) sum_sq += val * val;
    return std::sqrt(sum_sq);
  }
};
