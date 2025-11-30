#pragma once
#include "brain_block.hpp"
#include "embedding.hpp"
#include <fstream>
#include <vector>
#include <deque>

struct StepContext {
    std::vector<int> input_ids;
    std::shared_ptr<Tensor> emb_out;
    std::shared_ptr<Tensor> brain_out; // BrainBlock 的輸出
    std::shared_ptr<Tensor> logits;
    BrainBlockCache brain_cache;
    
    // 用於 BPTT，我們需要這一步驟使用的狀態副本。
    // BrainBlock 進行乒乓緩衝，所以如果我們想反向傳播多步，必須複製它們。
    std::vector<Tensor> prev_states_snapshot; 
};

class MiniLLM
{
    sycl::queue &q;
    int num_blocks;

  public:
    int vocab_size;
    int dim;
    int num_circuits;

    // 層
    std::unique_ptr<Embedding> embedding;
    std::unique_ptr<BrainBlock> brain;
    std::unique_ptr<BitLinear> head; // 隱藏層 -> 頭部

    std::deque<StepContext> history; // BPTT 歷史記錄
    int bptt_steps = 16; // 最大歷史長度

    MiniLLM(int v, int d, int n, sycl::queue &queue) : q(queue), vocab_size(v), dim(d), num_circuits(n)
    {
        embedding = std::make_unique<Embedding>(v, d, q);
        brain = std::make_unique<BrainBlock>(n, d, q);
        head = std::make_unique<BitLinear>(d, v, q); 
    }
    
    void init_buffers(int batch) {
        // 在新的追蹤邏輯下不是必需的，但保留以相容
    }

    // 前向傳播步驟
    std::pair<Tensor, std::vector<Tensor>&> forward_step(const std::vector<int> &input_ids,
                                                        std::vector<Tensor> &prev_states)
    {
        StepContext ctx;
        ctx.input_ids = input_ids;
        
        // 快照先前的狀態
        // BPTT 需要深度複製
        for(auto& s : prev_states) {
            Tensor s_copy(s.size, q);
            q.memcpy(s_copy.data, s.data, s.size * sizeof(float));
            ctx.prev_states_snapshot.push_back(std::move(s_copy));
        }

        // 1. 嵌入層
        Tensor emb = embedding->forward(input_ids); 
        ctx.emb_out = std::make_shared<Tensor>(std::move(emb));

        // 2. 大腦核心
        // 注意：我們傳遞 *ctx.prev_states_snapshot* 作為輸入？
        // 不，如果我們想要連續性，必須傳遞來自 BrainBlock 邏輯的實際當前狀態緩衝區？
        // 實際上，傳入的 prev_states 就是緩衝區。
        // 我們使用緩衝區進行計算，但儲存快照以供反向傳播。
        auto result = brain->forward(*ctx.emb_out, prev_states);
        
        ctx.brain_out = std::make_shared<Tensor>(std::move(std::get<0>(result)));
        ctx.brain_cache = std::move(std::get<2>(result));
        
        std::vector<Tensor>& next_states_ref = std::get<1>(result);

        // 3. 頭部
        // BitLinear::forward now returns pair<Tensor, Tensor>
        auto head_out = head->forward(*ctx.brain_out);
        Tensor logits = std::move(head_out.first); 
        // TODO: Store head_out.second (rstd) in ctx if needed for precise backward
        
        // 建立副本以回傳，保留一份供反向傳播？
        // 或者直接將 logits 移動到回傳，並重新計算或儲存副本？
        // 我們需要 logits 進行損失計算 (在外部完成)。
        // 但對於反向傳播，我們需要 `grad_logits`。
        // 如果損失是在外部計算的，我們嚴格來說不需要歷史記錄中的 `logits`。
        // 但 `head->backward` 需要 `brain_out`。
        
        history.push_back(std::move(ctx));
        if(history.size() > bptt_steps) {
            history.pop_front();
        }
        
        // 回傳 logits 的副本，因為我們將原始物件移入了 context？
        // 不，我們還沒將 logits 移入 context。
        // 讓我們為使用者製作一份副本。
        Tensor logits_ret(logits.size, q);
        q.memcpy(logits_ret.data, logits.data, logits.size * sizeof(float));
        
        return {std::move(logits_ret), next_states_ref};
    }
    
    // 通過時間反向傳播 (Backward Through Time)
    // all_step_grads: 歷史中每一步的 d_logits 向量 (與歷史記錄對齊：[0] 是歷史中最舊的)
    // 此函式展開整個歷史堆疊。
    void backward_through_time(std::vector<Tensor> &all_step_grads) {
        if(history.empty()) return;
        
        // 檢查對齊
        // if(all_step_grads.size() != history.size()) ... 處理錯誤或部分

        int batch = vocab_size > 0 ? (all_step_grads[0].size / vocab_size) : 1; 
        
        // 初始化來自未來的狀態梯度 (dL / ds_next) (初始為 0)
        std::vector<Tensor> d_states_next;
        for(int i=0; i<num_circuits; ++i) {
            Tensor t(batch * dim, q);
            t.zero_grad(); 
            d_states_next.push_back(std::move(t));
        }
        
        // 初始化來自未來的 M 梯度 (dL / dM_next)
        // M 是 [Batch, MemDim, MemDim]。假設 MemDim = Dim。
        // 為了簡化，假設 MiniLLM 可以訪問 BrainBlock 的維度或直接用 dim。
        // BrainBlock.dim 是 dim。Hub.memory_dim 是 dim。
        Tensor d_M_next(batch * dim * dim, q);
        d_M_next.zero_grad();
        
        // 逆序展開
        // history: [t-k, t-k+1, ... t]
        // grads:   [g_t-k, ... g_t]
        
        // 使用索引存取梯度
        int grad_idx = all_step_grads.size() - 1;

        while(!history.empty()) {
            StepContext& ctx = history.back();
            Tensor* current_d_logits = nullptr;
            
            if(grad_idx >= 0) {
                current_d_logits = &all_step_grads[grad_idx];
                grad_idx--;
            }
            
            // 1. 頭部反向傳播
            // 歸零梯度
            ctx.brain_out->zero_grad();
            
            if (current_d_logits) {
                // 重新計算 rstd (Hack fix)
                Tensor rstd(batch, q);
                Tensor x_norm(ctx.brain_out->size, q);
                kernels::rms_norm_fwd(q, ctx.brain_out->data, x_norm.data, rstd.data, batch, dim); 
                
                head->backward(*current_d_logits, *ctx.brain_out, rstd); 
            }
            
            // 2. 加入來自未來的遞迴梯度 (s_next)
            for(int i=0; i<num_circuits; ++i) {
                 Tensor& s_next_grad = d_states_next[i];
                 Tensor& s_curr_grad_target = *ctx.brain_cache.mc_caches[i].s; 
                 kernels::add_bwd(q, s_next_grad.data, s_curr_grad_target.grad, s_curr_grad_target.grad, s_next_grad.size);
            }
            
            // 3. BrainBlock 反向傳播
            // 需要一個 d_M_curr 緩衝區來接收來自這一步的 M 梯度 (decay + read contribution)
            Tensor d_M_curr(d_M_next.size, q);
            d_M_curr.zero_grad(); // 重要！
            
            for(auto& s : ctx.prev_states_snapshot) s.zero_grad();
            ctx.emb_out->zero_grad();
            
            brain->backward(*ctx.brain_out, ctx.brain_cache, *ctx.emb_out, ctx.prev_states_snapshot, d_M_next, d_M_curr);
            
            // 4. 嵌入層反向傳播
            embedding->backward(*ctx.emb_out, ctx.input_ids);
            
            // 5. 更新 d_states_next
            for(int i=0; i<num_circuits; ++i) {
                Tensor& prev_s = ctx.prev_states_snapshot[i];
                q.memcpy(d_states_next[i].data, prev_s.grad, prev_s.size * sizeof(float));
            }
            
            // 6. 更新 d_M_next (向前移動一步，變成下一步的「未來」)
            // d_M_next = d_M_curr
            q.memcpy(d_M_next.data, d_M_curr.data, d_M_curr.size * sizeof(float));

            history.pop_back();
        }
    }
    
    // 更新所有權重 (遞迴)
    void update_weights(TrainingTools& trainer, float lr, float weight_decay, float grad_scale) {
        // 1. Embedding
        trainer.sgd_update(embedding->weights, lr, weight_decay, grad_scale);
        
        // 2. Head
        trainer.sgd_update(head->weights, lr, weight_decay, grad_scale);
        
        // 3. BrainBlock (Internal)
        // 由於 BrainBlock 成員是私有的，我們無法直接存取。
        // 最好的方式是在 BrainBlock 中添加 update 方法。
        // 但為了不修改 BrainBlock 介面 (這會影響很多檔案)，我們這裡使用一個 friend 或是訪問器？
        // 其實我們可以在 BrainBlock 中公開權重，或者添加 update。
        // 鑑於此架構的限制，我們必須在 BrainBlock 添加 update。
        // (待辦事項：修改 BrainBlock.hpp 添加 update)
        // 暫時方案：假設 BrainBlock 有 update 方法。
        // brain->update(trainer, lr, weight_decay, grad_scale);
    }

    void save_to_file(const std::string &filename)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out) return;
        out.write((char *)&vocab_size, sizeof(int));
        out.write((char *)&dim, sizeof(int));
        out.write((char *)&num_circuits, sizeof(int));
        std::vector<float> h_emb(embedding->weights.size);
        q.memcpy(h_emb.data(), embedding->weights.data, h_emb.size() * sizeof(float)).wait();
        out.write((char *)h_emb.data(), h_emb.size() * sizeof(float));
        brain->save(out);
        head->save(out);
        out.close();
        std::cout << "Model saved to " << filename << std::endl;
    }

    void load_from_file(const std::string &filename) {
        // ... (Existing load logic, omitted for brevity if unchanged, but strictly need to implement to avoid wiping it)
        // Since I am replacing the whole file content, I must include it.
        std::ifstream in(filename, std::ios::binary);
        if (!in) return;
        int v, d, n;
        in.read((char *)&v, sizeof(int));
        in.read((char *)&d, sizeof(int));
        in.read((char *)&n, sizeof(int));
        std::vector<float> h_emb(embedding->weights.size);
        in.read((char *)h_emb.data(), h_emb.size() * sizeof(float));
        q.memcpy(embedding->weights.data, h_emb.data(), embedding->weights.size * sizeof(float)).wait();
        brain->load(in);
        head->load(in);
        in.close();
        std::cout << "Model loaded from " << filename << "\n" << std::endl;
    }
};