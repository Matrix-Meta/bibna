// src/cortical_hub.hpp
#pragma once
#include "bitlinear.hpp"
#include "common.hpp"
#include "kernels.hpp"
#include "training.hpp"

struct HubCache {
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> k_rstd;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> v_rstd;
    std::shared_ptr<Tensor> gate;
    std::shared_ptr<Tensor> gate_rstd;
    std::shared_ptr<Tensor> gate_sig;
    std::shared_ptr<Tensor> q_proj; // Query 投影
    std::shared_ptr<Tensor> q_rstd; // Query RMSNorm stats
    std::shared_ptr<Tensor> read_out; // M * q
};

class CorticalHub
{
    sycl::queue &q;
    int input_dim;
    int memory_dim;
    int num_circuits;

    // 投影層
    BitLinear w_k;
    BitLinear w_v;
    BitLinear w_gate;
    BitLinear w_q; // Query 投影

    // 參數
    float lambda = 0.95f;
    float eta = 0.1f;
    float clip = 5.0f;

  public:
    Tensor M_state; 

    CorticalHub(int in_dim, int mem_dim, int n_circuits, sycl::queue &queue)
        : q(queue), input_dim(in_dim), memory_dim(mem_dim), num_circuits(n_circuits), 
          w_k(in_dim, mem_dim, queue),
          w_v(in_dim, mem_dim, queue), 
          w_gate(in_dim, 1, queue),
          w_q(in_dim, mem_dim, queue), // 初始化 Query
          M_state(0, queue)
    {
    }

    void init_memory(int batch_size)
    {
        Tensor new_m(batch_size * memory_dim * memory_dim, q);
        q.fill(new_m.data, 0.0f, new_m.size);
        M_state = std::move(new_m);
    }

    // 前向傳播：讀取然後更新
    std::pair<Tensor, HubCache> forward(Tensor &inputs)
    {
        HubCache cache;
        int batch_total = inputs.size / input_dim; 
        
        // 1. 讀取 (Query)
        auto q_res = w_q.forward(inputs);
        cache.q_proj = std::make_shared<Tensor>(std::move(q_res.first));
        cache.q_rstd = std::make_shared<Tensor>(std::move(q_res.second));
        
        // 從 M 讀取
        Tensor read_out(inputs.size, q); 
        int batch = batch_total / num_circuits;
        kernels::memory_read_fwd(q, M_state.data, cache.q_proj->data, read_out.data, batch, num_circuits, memory_dim);
        
        // 2. 更新 (寫入)
        auto k_res = w_k.forward(inputs);
        cache.k = std::make_shared<Tensor>(std::move(k_res.first));
        cache.k_rstd = std::make_shared<Tensor>(std::move(k_res.second));
        
        auto v_res = w_v.forward(inputs);
        cache.v = std::make_shared<Tensor>(std::move(v_res.first));
        cache.v_rstd = std::make_shared<Tensor>(std::move(v_res.second));
        
        auto g_res = w_gate.forward(inputs);
        cache.gate = std::make_shared<Tensor>(std::move(g_res.first));
        cache.gate_rstd = std::make_shared<Tensor>(std::move(g_res.second));
        
        cache.gate_sig = std::make_shared<Tensor>(cache.gate->size, q);
        kernels::sigmoid_fwd(q, cache.gate->data, cache.gate_sig->data, cache.gate->size);

        // 就地更新 M
        kernels::fast_weight_update(q, M_state.data, cache.k->data, cache.v->data, cache.gate_sig->data, 
                                    batch, num_circuits, memory_dim, lambda, eta, clip);
                                    
        return {std::move(read_out), cache};
    }

    // 反向傳播
    void backward(Tensor &grad_read_out, Tensor &inputs, HubCache &cache, Tensor &d_M_future, Tensor &d_M_prev)
    {
        int batch_total = inputs.size / input_dim;
        int batch = batch_total / num_circuits;

        // 1. 記憶體讀取反向傳播
        Tensor d_q_proj(cache.q_proj->size, q);
        d_q_proj.zero_grad(); 
        kernels::memory_read_bwd_query(q, M_state.data, grad_read_out.data, d_q_proj.data, batch, num_circuits, memory_dim);
        kernels::memory_read_bwd_M(q, grad_read_out.data, cache.q_proj->data, d_M_prev.data, batch, num_circuits, memory_dim);

        // 2. 權重更新反向傳播
        Tensor d_k(cache.k->size, q); d_k.zero_grad();
        Tensor d_v(cache.v->size, q); d_v.zero_grad();
        Tensor d_gate_sig(cache.gate_sig->size, q); d_gate_sig.zero_grad();
        
        kernels::fast_weight_update_bwd(q, d_M_future.data, 
                                        cache.k->data, cache.v->data, cache.gate_sig->data,
                                        d_M_prev.data,
                                        d_k.data, d_v.data, d_gate_sig.data,
                                        batch, num_circuits, memory_dim, lambda, eta);

        // 3. 反向傳播到投影層
        w_q.backward(d_q_proj, inputs, *cache.q_rstd); 
        w_k.backward(d_k, inputs, *cache.k_rstd);
        w_v.backward(d_v, inputs, *cache.v_rstd);
        
        
        Tensor d_gate_logits(cache.gate->size, q);
        // 現在使用 sigmoid_bwd kernel
        kernels::sigmoid_bwd(q, d_gate_sig.data, cache.gate_sig->data, d_gate_logits.data, d_gate_logits.size);
        
        w_gate.backward(d_gate_logits, inputs, *cache.gate_rstd);
    }

    void update_memory(Tensor &inputs)
    {
        // Deprecated, merged into forward
    }
    float get_memory_norm() { return 0.0f; } // Placeholder

    void save(std::ofstream &out)
    {
        w_k.save(out);
        w_v.save(out);
        w_gate.save(out);
        w_q.save(out);
    }

    void load(std::ifstream &in)
    {
        w_k.load(in);
        w_v.load(in);
        w_gate.load(in);
        w_q.load(in);
    }
    
    // 更新權重
    void update(TrainingTools &trainer, float lr, float decay, float scale) {
        w_k.weights.zero_grad(); // 暫時不學習 K, V, Gate (因為沒有反向傳播)
        w_v.weights.zero_grad();
        w_gate.weights.zero_grad();
        // 僅更新 Query (因為只有它有正確的梯度)
        trainer.sgd_update(w_q.weights, lr, decay, scale);
    }
};
