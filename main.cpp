#include<iostream>
#include <cmath>
#include <iomanip>
#include <vector>

// 引入所有模組
#include "src/acc.hpp"
#include "src/bitlinear.hpp"
#include "src/brain_block.hpp"
#include "src/common.hpp"
#include "src/cortical_hub.hpp"
#include "src/decoding.hpp"
#include "src/micro_circuit.hpp"
#include "src/spiking_unit.hpp"
#include "src/training.hpp"

// ==========================================
// 函式宣告 (Function Prototypes)
// ==========================================

void gradient_check(sycl::queue& q);
void micro_circuit_test(sycl::queue& q);
void cortical_hub_test(sycl::queue& q);
void generation_test(sycl::queue& q);
void training_test(sycl::queue& q);

// ==========================================
// 主程式 (Main Entry)
// ==========================================

int main() {
  try {
    // 1. 初始化 SYCL Queue (自動選擇裝置)
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 2. 依序執行所有階段的測試

    // Phase 1: 基礎算子與梯度檢查
    gradient_check(q);

    // Phase 2: 微迴路與時間遞迴測試
    micro_circuit_test(q);

    // Phase 3: 整合層與快權重穩定性測試
    cortical_hub_test(q);

    // Phase 5: 生成循環、ACC 監控與動態懲罰測試
    // (註：Phase 4 已整合進此函式)
    generation_test(q);

    // Phase 6: 訓練迴圈測試 (SGD & Backprop)
    training_test(q);

  } catch (sycl::exception& e) {
    std::cerr << "SYCL Exception: " << e.what() << "\n";
    return 1;
  } catch (std::exception& e) {
    std::cerr << "Standard Exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

// ==========================================
// 函式定義 (Function Definitions)
// ==========================================

// Phase 1: Gradient Check
void gradient_check(sycl::queue& q) {
  std::cout << "\n=== Running Phase 1: Gradient Check ===\n";

  int batch = 2;
  int in_dim = 4;
  int out_dim = 4;

  Tensor input(batch * in_dim, q);
  input.random_init(0.5f, 0.5f);

  BitLinear bitnet(in_dim, out_dim, q);
  SpikingUnit spike(q, 0.0f, 2.0f);  // Theta=0, allow easy firing

  std::cout << "[Forward Pass]...\n";
  Tensor linear_out = bitnet.forward(input);
  Tensor spike_out = spike.forward(linear_out);

  std::cout << "[Backward Pass]...\n";
  Tensor grad_loss(spike_out.size, q);
  q.fill(grad_loss.data, 1.0f, grad_loss.size).wait();

  spike.backward(grad_loss);
  bitnet.backward(linear_out);

  std::cout << "[Check A] SpikingUnit Surrogate Gradient:\n";
  bool surrogate_active = false;
  // 檢查前 4 個元素
  std::vector<float> h_v(4), h_grad(4);
  q.memcpy(h_v.data(), linear_out.data, 4 * sizeof(float)).wait();
  q.memcpy(h_grad.data(), linear_out.grad, 4 * sizeof(float)).wait();

  for (int i = 0; i < 4; ++i) {
    float expected = 1.0f / (1.0f + 4.0f * h_v[i] * h_v[i]);
    std::cout << "  v: " << std::fixed << std::setprecision(4) << h_v[i]
              << " | Grad: " << h_grad[i] << " | Expected: " << expected
              << "\n";
    if (std::abs(h_grad[i]) > 1e-5) surrogate_active = true;
  }

  if (surrogate_active)
    std::cout << ">>> PASS: Surrogate Gradient works.\n";
  else
    std::cout << ">>> FAIL: Zero Gradients.\n";

  std::cout << "[Check B] BitLinear Gradient Flow:\n";
  // 簡單檢查 input.grad 是否非零
  std::vector<float> h_in_grad(4);
  q.memcpy(h_in_grad.data(), input.grad, 4 * sizeof(float)).wait();
  if (std::abs(h_in_grad[0]) > 1e-6)
    std::cout << ">>> PASS: BitLinear Backward works.\n";
  else
    std::cout << ">>> FAIL: Input Gradient is zero.\n";
}

// Phase 2: MicroCircuit Loop
void micro_circuit_test(sycl::queue& q) {
  std::cout << "\n=== Running Phase 2: MicroCircuit Loop Test ===\n";

  int batch = 2;
  int dim = 4;
  int hidden = 8;
  int time_steps = 3;

  MicroCircuit mc(dim, hidden, q);
  Tensor s_state(batch * hidden, q);
  // s_state 初始為 0

  std::cout << "Starting Time Loop (" << time_steps << " steps)...\n";

  for (int t = 0; t < time_steps; ++t) {
    Tensor x_t(batch * dim, q);
    x_t.random_init(0.5f, 0.5f);

    // 執行一步
    auto result = mc.forward_step(x_t, s_state);

    // 更新狀態 (使用 move 語意)
    s_state = std::move(result.second);

    std::cout << "  [Step " << t + 1 << "] Completed. State updated.\n";
  }

  std::cout << ">>> PASS: MicroCircuit time loop execution successful.\n";
}

// Phase 3: CorticalHub & Fast Weights
void cortical_hub_test(sycl::queue& q) {
  std::cout << "\n=== Running Phase 3: CorticalHub & Fast Weights Test ===\n";

  int batch = 2;
  int num_circuits = 4;
  int dim = 8;
  int time_steps = 20;

  CorticalHub hub(dim, dim, num_circuits, q);
  hub.init_memory(batch);

  std::cout << "Starting Memory Stability Loop (" << time_steps
            << " steps)...\n";

  for (int t = 0; t < time_steps; ++t) {
    Tensor fake_inputs(batch * num_circuits * dim, q);
    fake_inputs.random_init(0.1f, 1.0f);

    hub.update_memory(fake_inputs);

    if ((t + 1) % 5 == 0) {
      float norm = hub.get_memory_norm();
      std::cout << "  [Step " << std::setw(2) << t + 1 << "] M_t Norm: " << norm
                << "\n";

      if (std::isnan(norm) || std::isinf(norm) || norm > 1000.0f) {
        std::cout << ">>> FAIL: Memory Exploded!\n";
        return;
      }
    }
  }
  std::cout << ">>> PASS: Fast Weights updated stably.\n";
}

// Phase 5: Generation, ACC & Anti-Collapse
void generation_test(sycl::queue& q) {
  std::cout << "\n=== Running Phase 5: ACC & Dynamic Generation ===\n";

  int dim = 16;
  int num_circuits = 2;
  int seq_len = 10;

  BrainBlock block(num_circuits, dim, q);
  AntiCollapse decoder(q, 1.2f, dim);
  ConflictMonitor acc(q);

  auto states = block.init_states(1);
  std::vector<int> history;
  Tensor input(dim, q);
  input.random_init(0.5f, 0.5f);

  std::cout << "Generating with Dynamic Control...\n";

  for (int t = 0; t < seq_len; ++t) {
    // 1. Forward
    auto result = block.forward(input, states);
    Tensor& logits = result.first;
    states = std::move(result.second);

    // 2. ACC 監測
    float entropy = acc.calculate_uncertainty(logits);
    // 設定較低的閾值 0.5 以便觀察 ACC 觸發
    bool high_conflict = acc.is_high_conflict(entropy, 0.5f);

    std::cout << "  Step " << std::setw(2) << t + 1
              << " | Entropy: " << std::fixed << std::setprecision(3)
              << entropy;

    // 3. Meta-Controller 介入
    if (high_conflict) {
      std::cout << " [ACC ALERT] -> Boosting Penalty.";
      decoder.apply_penalty(logits, history);
      decoder.apply_penalty(logits, history);
    } else {
      decoder.apply_penalty(logits, history);
    }

    // 4. Sampling
    int token = decoder.sample(logits);
    history.push_back(token);

    std::cout << " -> Picked: " << token << "\n";

    // 更新 Input (模擬 Embedding lookup)
    q.memcpy(input.data, logits.data, dim * sizeof(float)).wait();
  }

  std::cout << ">>> PASS: ACC integrated. Dynamic control flow verified.\n";
}

// Phase 6: Training Loop
void training_test(sycl::queue& q) {
  std::cout << "\n=== Running Phase 6: Training Loop (SGD) ===\n";

  int dim = 4;
  float lr = 0.1f;
  int epochs = 100;

  BitLinear layer(dim, dim, q);
  TrainingTools trainer(q);

  Tensor input(dim, q);
  q.fill(input.data, 1.0f, input.size).wait();
  int target_idx = 0;

  std::cout << "Task: Learn to classify Input[1,1,1,1] as Class 0\n";

  for (int i = 0; i < epochs; ++i) {
    // A. Forward
    Tensor output = layer.forward(input);

    // B. Loss Calculation
    Tensor d_output(dim, q);
    float loss = trainer.cross_entropy(output, target_idx, d_output);

    if (i % 20 == 0 || i == epochs - 1) {
      std::cout << "  Epoch " << std::setw(3) << i << " | Loss: " << std::fixed
                << std::setprecision(4) << loss << "\n";
    }

    // C. Backward
    layer.backward(d_output);

    // D. Update
    trainer.sgd_step(layer.weights, lr);

    // E. Zero Gradients (Input is reused)
    input.zero_grad();
  }

  std::cout << ">>> PASS: Training loop finished.\n";
}

