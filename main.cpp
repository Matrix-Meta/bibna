#include "src/acc.hpp"
#include "src/bitlinear.hpp"
#include "src/brain_block.hpp"
#include "src/common.hpp"
#include "src/cortical_hub.hpp"
#include "src/decoding.hpp"
#include "src/embedding.hpp"
#include "src/micro_circuit.hpp"
#include "src/mini_llm.hpp"
#include "src/spiking_unit.hpp"
#include "src/tokenizer.hpp"
#include "src/training.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

int main()
{
    try
    {
        sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
        std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // ==========================================
        // 1. 資料準備 (Data Preparation)
        // ==========================================
        std::cout << "Preparing Dataset...\n";
        std::string base_text = "Hello, World! 現在是2025年，類腦脈衝神經網路模型BiBNA的極速訓練測試。";
        std::string train_text;
        // 重複文本以產生足夠的 Token 進行批次訓練
            for (int i = 0; i < 5000; ++i)
            train_text += base_text + " ";

        SimpleTokenizer tokenizer;
        tokenizer.build_vocab(train_text); // 使用一部分來建立詞彙表
        
        std::vector<int> full_dataset = tokenizer.encode(train_text);
        std::cout << "Total Tokens: " << full_dataset.size() << "\n";
        std::cout << "Vocab Size: " << tokenizer.vocab_size << "\n";

        // ==========================================
        // 2. 超參數 (Hyperparameters)
        // ==========================================
        int dim = 64;
        int num_circuits = 8; // 增加電路數量
        int batch_size = 128; // 極大批次
        int seq_len = 128;    // 更長的 BPTT 視窗
        float lr = 0.002f;    // 稍微降低 LR 以適應大批次
        int epochs = 5;       // 減少 Epochs 因為數據量變大
        
        // 計算批次數量
        // 我們將數據切分為 (Batch_Size, Num_Batches * Seq_Len) 的形狀?
        // 簡單做法：滑動視窗或直接切塊。
        // 這裡採用標準語言模型做法：將數據視為長序列，切分為 BatchSize 個片段。
        // 每個片段長度 = Total / BatchSize
        int total_len = full_dataset.size();
        int batch_len = total_len / batch_size; // 每個批次序列的長度
        int num_steps = batch_len / seq_len;    // 每個批次序列可以切多少個 seq_len
        
        std::cout << "Training Config:\n"
                  << "  Batch Size: " << batch_size << "\n"
                  << "  Seq Len   : " << seq_len << "\n"
                  << "  Batches/Ep: " << num_steps << "\n"
                  << "  Total Step: " << num_steps * epochs << "\n" << std::endl;

        // 準備 Batch 數據指標
        // data_batches[b] 指向第 b 個序列的起始位置
        std::vector<int> data_starts(batch_size);
        for(int b=0; b<batch_size; ++b) {
            data_starts[b] = b * batch_len;
        }

        // ==========================================
        // 3. 模型初始化
        // ==========================================
        MiniLLM model(tokenizer.vocab_size, dim, num_circuits, q);
        model.init_buffers(batch_size); 
        
        TrainingTools trainer(q);
        AntiCollapse decoder(q, 1.2f, dim);

        // ==========================================
        // 4. 訓練迴圈 (Training Loop)
        // ==========================================
        std::cout << "=== Starting High-Performance Training ===\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            // 每個 Epoch 開始時重置狀態 (SNN/RNN 狀態)
            std::vector<Tensor>* states_ptr = &model.brain->init_states(batch_size);
            
            float epoch_loss = 0.0f;
            int loss_count = 0;

            // 批次迴圈 (實際上是長序列的時間步迴圈，只是被切分為 seq_len 塊)
            for (int step = 0; step < num_steps; ++step)
            {
                // 為了避免複製，我們需要 Tensor 支援移動，std::vector 支援 push_back(move)
                // 但我們在 loop 內，d_logits 是區域變數。
                // 我們需要一個持久的容器給 backward_through_time 使用。
                // 由於 MiniLLM 的設計是 history 驅動的，我們只需傳遞 d_logits。
                // 但 backward_through_time 需要**所有步驟**的梯度。
                // 所以我們需要在外部收集一個 `window_grads`。
                static std::vector<Tensor> window_grads; // Static to reuse memory capacity
                window_grads.clear();

                // 序列內的 Token 迴圈 (BPTT Window)
                for (int t = 0; t < seq_len; ++t)
                {
                    // 準備當前時間步的輸入與目標
                    // Input: [Batch] -> indices
                    std::vector<int> input_ids(batch_size);
                    std::vector<int> target_ids(batch_size);
                    
                    int global_ptr_offset = step * seq_len + t;
                    
                    for(int b=0; b<batch_size; ++b) {
                        int ptr = data_starts[b] + global_ptr_offset;
                        if (ptr < total_len - 1) {
                            input_ids[b] = full_dataset[ptr];
                            target_ids[b] = full_dataset[ptr+1];
                        } else {
                            input_ids[b] = 0; // Padding/End
                            target_ids[b] = 0;
                        }
                    }

                    // 前向傳播 (Parallel Batch)
                    auto result = model.forward_step(input_ids, *states_ptr);
                    Tensor &logits = result.first;
                    states_ptr = &result.second;

                    // 損失計算 & 梯度
                    Tensor d_logits(logits.size, q);
                    float loss = trainer.cross_entropy_loss(logits, target_ids, d_logits, t, 1000); // 減少同步頻率
                    
                    if (loss > 0.0f) {
                        epoch_loss += loss;
                        loss_count++;
                    }
                    
                    // 儲存 BPTT 梯度
                    window_grads.push_back(std::move(d_logits));
                } // End Seq Loop

                // 觸發 BPTT 和 權重更新 (每 seq_len 步一次)
                model.backward_through_time(window_grads);
                model.update_weights(trainer, lr, 1e-4f, 1.0f / (seq_len * batch_size));
                
                // 進度條
                if (step % 10 == 0) {
                    std::cout << "\rEpoch " << epoch << " | Step " << step << "/" << num_steps 
                              << " | Loss: " << (loss_count > 0 ? epoch_loss/loss_count : 0.0f) << std::flush;
                    epoch_loss = 0.0f;
                    loss_count = 0;
                }
            }
            std::cout << "\n";
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        std::cout << "Training Finished in " << diff.count() << " s\n";
        std::cout << "Speed: " << (total_len * epochs) / diff.count() << " tokens/s\n";

        // =========================================
        // 5. 存檔與推論 (Inference)
        // =========================================
        model.save_to_file("BiBNA_Batched.bin");
        tokenizer.save_vocab("vocab.txt"); // Ensure this line is present
        
        std::cout << "\n=== Batched Inference Test ===\n";
        // 為了推論，我們使用 Batch=1
        MiniLLM infer_model(tokenizer.vocab_size, dim, num_circuits, q);
        infer_model.load_from_file("BiBNA_Batched.bin");
        
        std::string prompt = "Hello";
        std::vector<int> input = tokenizer.encode(prompt);
        std::vector<Tensor>* s_ptr = &infer_model.brain->init_states(1);
        
        std::cout << prompt;
        int next_token = input.back();
        
        for(int i=0; i<50; ++i) {
            auto res = infer_model.forward_step({next_token}, *s_ptr);
            s_ptr = &res.second;
            
            // Sample
            next_token = decoder.sample(res.first);
            std::cout << tokenizer.decode({next_token}) << std::flush;
        }
        std::cout << "\nDone.\n";
    }
    catch (sycl::exception const &e)
    {
        std::cout << "SYCL Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
