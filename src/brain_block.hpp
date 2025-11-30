#pragma once
#include "cortical_hub.hpp"
#include "micro_circuit.hpp"
#include <fstream>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

struct BrainBlockCache {
    std::vector<MicroCircuitCache> mc_caches;
    HubCache hub_cache;
    std::shared_ptr<Tensor> hub_input; // [B*N, D]
};

class BrainBlock
{
    sycl::queue &q;
    int num_circuits;
    int dim;
    std::vector<std::unique_ptr<MicroCircuit>> circuits;
    CorticalHub hub;

    std::vector<std::vector<Tensor>> state_buffers;
    int current_buffer_idx = 0;

  public:
    BrainBlock(int n_circuits, int d, sycl::queue &queue)
        : q(queue), num_circuits(n_circuits), dim(d), hub(d, d, n_circuits, queue)
    {
        for (int i = 0; i < num_circuits; ++i)
        {
            circuits.push_back(std::make_unique<MicroCircuit>(d, d, queue));
        }
        state_buffers.resize(2);
    }

    std::vector<Tensor>& init_states(int batch)
    {
        hub.init_memory(batch);
        for(auto& mc : circuits) {
            mc->init_buffers(batch);
        }
        for(int buf = 0; buf < 2; ++buf) {
            state_buffers[buf].clear();
            state_buffers[buf].reserve(num_circuits);
            for (int i = 0; i < num_circuits; ++i)
            {
                Tensor s(batch * dim, q);
                q.fill(s.data, 0.0f, s.size);
                state_buffers[buf].push_back(std::move(s));
            }
        }
        current_buffer_idx = 0;
        return state_buffers[current_buffer_idx];
    }

    // 修改後的前向傳播：回傳快取
    std::tuple<Tensor, std::vector<Tensor>&, BrainBlockCache> forward(Tensor &input, std::vector<Tensor> &current_states)
    {
        BrainBlockCache block_cache;
        int batch = input.size / dim;
        
        int next_buffer_idx = 1 - current_buffer_idx;
        std::vector<Tensor>& next_states_buf = state_buffers[next_buffer_idx];
        
        // 1. 微電路前向傳播
        std::vector<float*> mc_output_ptrs;
        mc_output_ptrs.reserve(num_circuits);

        for (int i = 0; i < num_circuits; ++i)
        {
            auto [mc_out, next_s, mc_cache] = circuits[i]->forward_step_traced(input, current_states[i]);
            
            // 移動/複製下一個狀態到緩衝區
            // 由於 `next_s` 是一個區域 Tensor，如果我們重新賦值，可以將其移動到緩衝區嗎？
            // 但緩衝區是固定的。我們複製數據。
            Tensor& target = next_states_buf[i];
            q.memcpy(target.data, next_s.data, target.size * sizeof(float));
            
            // 我們需要保留 `mc_out` 以供 Hub 輸入使用。
            // 它是一個 Tensor。我們需要聚合它們。
            // 我們不能只儲存指向堆疊變數的指標。
            // 讓我們分配 `hub_input` 來串接它們。
            // 我們在步驟 2 中做這件事。但 `mc_out` 在此迴圈迭代後會被銷毀？
            // 等等，`mc_out` 是從 `forward_step_traced` 按值回傳的。
            // 我們必須立即儲存或複製它。
            // 優化：直接複製到 `hub_input`。
            
            block_cache.mc_caches.push_back(mc_cache);
            // 我們需要一個暫時的地方讓 mc_outputs 存活直到 Hub 輸入建構？
            // 實際上我們在下面建構 Hub 輸入。
            // 讓我們使用 lambda 立即複製？不，平行複製比較好。
            // 我們需要讓 `mc_out` 存活。
            // 這個設計在複製上稍微沒效率，但安全。
            // 更好：`hub_input` 就是 MC 輸出的緩衝區。
        }
        
        // 2. 準備 Hub 輸入 (串接 MC 輸出)
        Tensor hub_input(batch * num_circuits * dim, q);
        size_t chunk_size = batch * dim;
        
        // 重新執行迴圈以填充 hub_input (需要改變迴圈結構)
        // 理想情況下我們傳遞 `hub_input` 的切片給 MC 前向傳播？
        // 目前，我們只是重新執行邏輯或快取輸出。
        // 讓我們暫時將輸出快取在 shared_ptrs 向量中？
        std::vector<std::shared_ptr<Tensor>> temp_outs;
        block_cache.mc_caches.clear(); // 先清除

        for (int i = 0; i < num_circuits; ++i)
        {
            auto [mc_out, next_s, mc_cache] = circuits[i]->forward_step_traced(input, current_states[i]);
            block_cache.mc_caches.push_back(mc_cache);
            
            // 更新狀態
            q.memcpy(next_states_buf[i].data, next_s.data, next_s.size * sizeof(float));
            
            // 複製到 Hub 輸入
            q.memcpy(hub_input.data + i * chunk_size, mc_out.data, chunk_size * sizeof(float));
        }
        
        block_cache.hub_input = std::make_shared<Tensor>(std::move(hub_input));

        // 3. Hub 前向傳播 (讀取 + 寫入)
        auto [hub_out, h_cache] = hub.forward(*block_cache.hub_input);
        block_cache.hub_cache = h_cache;

        // 4. 聚合輸出 (Hub 讀取輸出 + MC 輸出？或者只是 Hub 讀取？)
        // "BrainBlock" 通常意味著 Hub 整合資訊。
        // 如果 Hub 讀取是 [B*N, D]，我們可以將其加總。
        // 讓我們加總 Hub 讀取輸出。
        
        Tensor output(batch * dim, q);
        q.fill(output.data, 0.0f, output.size);
        
        // 加總歸約核心
        // hub_out 是展平的 [B, N, D]。
        int n_circuits = num_circuits; // 局部變數以避免捕獲 this
        int block_dim = dim; // 局部變數以避免捕獲 this
        q.parallel_for(sycl::range<2>(batch, dim), [=, h_ptr = hub_out.data, out_ptr = output.data](sycl::id<2> idx) {
             int b = idx[0];
             int d = idx[1];
             float sum = 0.0f;
             for(int i=0; i<n_circuits; ++i) {
                 sum += h_ptr[b * n_circuits * block_dim + i * block_dim + d];
             }
             out_ptr[b * block_dim + d] = sum / n_circuits;
        });

        current_buffer_idx = next_buffer_idx;
        return {std::move(output), next_states_buf, block_cache};
    }
    
    // 反向傳播
    // grad_output: dL/dy
    // input: 原始輸入 [B, Dim] (需要 .grad)
    // current_states: 前向傳播使用的狀態 (需要 .grad)
    // d_M_future: 來自未來的 M 梯度 (輸入)
    // d_M_prev: 累積此步驟的 M 梯度 (輸出)
    void backward(Tensor &grad_output, BrainBlockCache &cache, Tensor &input, std::vector<Tensor> &current_states, Tensor &d_M_future, Tensor &d_M_prev) {
        // 1. 分發梯度到 Hub 輸出
        int batch = grad_output.size / dim;
        Tensor d_hub_out(batch * num_circuits * dim, q);
        
        int n_circuits = num_circuits; // 局部變數
        int block_dim = dim; // 局部變數
        q.parallel_for(sycl::range<2>(batch, dim), [=, gout = grad_output.data, dhub = d_hub_out.data](sycl::id<2> idx) {
            int b = idx[0];
            int d = idx[1];
            float val = gout[b*block_dim + d] / n_circuits;
            for(int i=0; i<n_circuits; ++i) {
                dhub[b*n_circuits*block_dim + i*block_dim + d] = val;
            }
        });
        
        // 2. Hub 反向傳播
        // 傳遞 d_M_future 和 d_M_prev
        cache.hub_input->zero_grad();
        hub.backward(d_hub_out, *cache.hub_input, cache.hub_cache, d_M_future, d_M_prev);
        
        // 3. 微電路反向傳播
        size_t chunk_size = batch * dim;
        
        for(int i=0; i<num_circuits; ++i) {
             Tensor d_mc_out(batch * dim, q);
             q.memcpy(d_mc_out.data, cache.hub_input->grad + i * chunk_size, chunk_size * sizeof(float));
             circuits[i]->backward(d_mc_out, cache.mc_caches[i], input, current_states[i]);
        }
    }

    void save(std::ofstream &out)
    {
        hub.save(out);
        for (auto &circuit : circuits)
        {
            circuit->save(out);
        }
    }

    void load(std::ifstream &in)
    {
        hub.load(in);
        for (auto &circuit : circuits)
        {
            circuit->load(in);
        }
    }
    
    // 更新所有內部權重
    void update(TrainingTools &trainer, float lr, float decay, float scale) {
        hub.update(trainer, lr, decay, scale);
        for(auto& mc : circuits) {
            mc->update(trainer, lr, decay, scale);
        }
    }
};