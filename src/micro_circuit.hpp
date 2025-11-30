// src/micro_circuit.hpp
#pragma once
#include <utility>

#include "bitlinear.hpp"
#include "spiking_unit.hpp"
#include "training.hpp"

struct MicroCircuitCache {
    std::shared_ptr<Tensor> u_in;  // W_in * x
    std::shared_ptr<Tensor> u_in_rstd; 
    std::shared_ptr<Tensor> u_rec; // W_rec * s_prev
    std::shared_ptr<Tensor> u_rec_rstd;
    std::shared_ptr<Tensor> v;     // 電位 (Potential)
    std::shared_ptr<Tensor> s;     // 脈衝狀態 (Spike state)
    std::shared_ptr<Tensor> proj;  // W_out * s
    std::shared_ptr<Tensor> proj_rstd;
    // 反向傳播也需要輸入指標 (x_t, s_prev_t)。
    // 呼叫者 (MiniLLM) 應負責管理 x_t 和 s_prev_t 的生命週期。
};

class MicroCircuit
{
    sycl::queue &q;
    int input_dim;
    int hidden_dim;

    // 組件
    BitLinear w_in;
    BitLinear w_rec;
    BitLinear w_out;
    SpikingUnit spikes;

  public:
    MicroCircuit(int dim, int hidden, sycl::queue &queue)
        : q(queue), input_dim(dim), hidden_dim(hidden), w_in(dim, hidden, queue), w_rec(hidden, hidden, queue),
          w_out(hidden, dim, queue), 
          spikes(queue, 1.0f, 2.0f)
    {
    }

    void init_buffers(int batch) {
        // 已棄用，改用每步分配或由外部管理 BPTT
    }

    ~MicroCircuit() {}

    // 前向傳播並回傳快取
    // x_t: 輸入 [B, Dim]
    // s_prev_t: 上一步的脈衝狀態 [B, Hidden]
    // 回傳: {輸出 [B, Dim], 新脈衝狀態 [B, Hidden], 快取}
    std::tuple<Tensor, Tensor, MicroCircuitCache> forward_step_traced(Tensor &x_t, Tensor &s_prev_t)
    {
        MicroCircuitCache cache;
        
        // 1. 計算電流 (為追蹤分配新 Tensor)
        // 注意：為了高效能，我們應該使用記憶體池。目前每步使用 malloc_shared。
        auto u_in_res = w_in.forward(x_t);
        cache.u_in = std::make_shared<Tensor>(std::move(u_in_res.first));
        cache.u_in_rstd = std::make_shared<Tensor>(std::move(u_in_res.second));
        
        auto u_rec_res = w_rec.forward(s_prev_t);
        cache.u_rec = std::make_shared<Tensor>(std::move(u_rec_res.first));
        cache.u_rec_rstd = std::make_shared<Tensor>(std::move(u_rec_res.second));
        
        int batch = x_t.size / input_dim;
        cache.v = std::make_shared<Tensor>(batch * hidden_dim, q);
        cache.s = std::make_shared<Tensor>(batch * hidden_dim, q);
        
        // 2. 融合: 加法 + 激發脈衝
        kernels::fused_add_spike(q, cache.u_in->data, cache.u_rec->data, cache.v->data, cache.s->data, cache.v->size, spikes.theta);

        // 3. 輸出投影
        auto proj_res = w_out.forward(*cache.s);
        cache.proj = std::make_shared<Tensor>(std::move(proj_res.first));
        cache.proj_rstd = std::make_shared<Tensor>(std::move(proj_res.second));
        
        // 4. 殘差相加
        Tensor out(batch * input_dim, q);
        kernels::fused_residual_add(q, x_t.data, cache.proj->data, out.data, out.size);
        
        // 回傳輸出 (移動), 新狀態 (複製 s, 或分享指標邏輯?), 和快取
        // 我們回傳 s 的副本做為「新狀態 Tensor」，因為快取中的那個屬於這一步的歷史
        Tensor s_new(cache.s->size, q);
        q.memcpy(s_new.data, cache.s->data, s_new.size * sizeof(float)); // 移除了 wait，依賴順序屬性
        
        return {std::move(out), std::move(s_new), cache};
    }

    // 反向傳播
    // grad_output: dL/d(out)
    // cache: 前向傳播儲存的快取
    // x_t, s_prev_t: 前向傳播使用的輸入 (需要存活且有 .grad 緩衝區)
    // 回傳: {dL/dx, dL/ds_prev} (累積到 x_t.grad 和 s_prev_t.grad)
    void backward(Tensor &grad_output, MicroCircuitCache &cache, Tensor &x_t, Tensor &s_prev_t) {
        // 1. 殘差相加反向傳播
        // out = x + proj
        // dx += dout, dproj += dout
        // proj 是中間產物，所以我們需要它的梯度緩衝區。
        // cache.proj->grad 需要先歸零嗎？快取 Tensor 是在前向建立的，所以梯度是 0。
        
        // 複製 grad_output 到 x_t.grad (累積? 不，通常這是此區塊反向傳播的開始)
        // 但 x_t 可能被其他區塊使用？在此架構中，輸入來自 Embedding 或前一個區塊。
        // 我們應該 累積 到 x_t.grad。
        kernels::add_bwd(q, grad_output.data, x_t.grad, cache.proj->grad, x_t.size);
        
        // 2. w_out 反向傳播
        // proj = w_out * s
        // dL/ds += w_out.backward(dproj)
        // s 在快取中。
        w_out.backward(*cache.proj, *cache.s, *cache.proj_rstd); // 更新 w_out.grad 和 cache.s->grad
        
        // 3. 脈衝單元反向傳播
        // s = spike(v)
        // dv = ds * spike' (surrogate)
        spikes.backward(*cache.s, *cache.v); // 更新 cache.v->grad
        
        // 4. 融合加法反向傳播 (v = u_in + u_rec)
        // du_in = dv, du_rec = dv
        kernels::add_bwd(q, cache.v->grad, cache.u_in->grad, cache.u_rec->grad, cache.v->size);
        
        // 5. 線性層反向傳播
        // u_in = w_in * x
        w_in.backward(*cache.u_in, x_t, *cache.u_in_rstd); // 累積到 x_t.grad
        
        // u_rec = w_rec * s_prev
        w_rec.backward(*cache.u_rec, s_prev_t, *cache.u_rec_rstd); // 累積到 s_prev_t.grad
    }

    void save(std::ofstream &out)
    {
        w_in.save(out);
        w_rec.save(out);
        w_out.save(out);
    }

    void load(std::ifstream &in)
    {
        w_in.load(in);
        w_rec.load(in);
        w_out.load(in);
    }
    
    // 更新權重
    void update(TrainingTools &trainer, float lr, float decay, float scale) {
        trainer.sgd_update(w_in.weights, lr, decay, scale);
        trainer.sgd_update(w_rec.weights, lr, decay, scale);
        trainer.sgd_update(w_out.weights, lr, decay, scale);
    }
    
    // Legacy support / Placeholder
    Tensor& get_current_state() {
        static Tensor dummy(0, q);
        return dummy;
    }
    Tensor& forward_step(Tensor &x_t, Tensor &s_prev_t) {
        // This function is deprecated by trace logic, but kept to avoid build breaks if main not updated yet.
        // It will fail if called.
        throw std::runtime_error("Old forward_step called in MicroCircuit");
    }
};
