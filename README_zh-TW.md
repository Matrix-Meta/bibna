# bibna (Brain-Inspired BitNet Architecture)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![SYCL](https://img.shields.io/badge/Framework-SYCL%20(DPC%2B%2B)-orange.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)

> **注意：** 本專案具有高度實驗性質，目前正處於積極的原型開發階段。

[English](README.md)

---

## 📖 專案概述

**bibna** 是一個基於 C++ 與 SYCL 實作的前沿研究框架。它的目標是將 **BitNet b1.58**（1.58-bit 大型語言模型）的極致運算效率，與 **脈衝神經網絡 (SNNs)** 及 **Hebb 學習 (Hebbian Learning)** 的生物合理性相結合。

與將神經網絡視為靜態浮點矩陣運算的傳統深度學習框架不同，`bibna` 從底層重新構建了計算堆疊，以支援：
1.  **三元權重 (Ternary Weights `{-1, 0, +1}`)**：最小化記憶體頻寬需求與運算能耗。
2.  **事件驅動計算 (Event-Driven Computation)**：使用具有內部狀態（膜電位）的脈衝神經元。
3.  **動態記憶 (Dynamic Memory)**：實作「快權重 (Fast Weights)」，在推理過程中即時更新，提供工作記憶 (Working Memory) 上下文。

## 🧠 核心理念

本架構的設計基於一個假設：通用智能不僅僅需要靜態的模式匹配，更需要：
*   **局部推理 (Local Reasoning)**：由 `MicroCircuit`（微電路）處理。這是包含遞歸連接的小型神經元集群，負責時序資訊的局部處理。
*   **全局整合 (Global Integration)**：由 `CorticalHub`（皮層樞紐）管理，負責整合局部訊號並透過注意力機制進行調變。
*   **自我修正 (Self-Correction)**：一個 **ACC (前扣帶迴)** 模組，用於偵測衝突與不確定性，在輸出回應前觸發「反思」迴路。

## 🏗️ 架構與組件

本專案遵循嚴格的「自底向上」層級設計：

### 1. 底層原語 (`src/kernels.hpp`, `src/bitlinear.hpp`)
*   **BitLinear**：基礎的全連接層。它使用動態量化 (On-the-fly quantization) 將 FP32 激活值投影到三元權重上。
*   **直通估計器 (STE)**：允許梯度在訓練過程中穿過不可微分的量化步驟。

### 2. 神經元模型 (`src/spiking_unit.hpp`)
*   **SpikingUnit**：取代了傳統的 ReLU/GELU。神經元會隨時間累積「電壓」，並在超過閾值時發放離散脈衝。
*   **代理梯度 (Surrogate Gradients)**：解決了脈衝不可微分的問題，使脈衝神經元支援時間反向傳播 (BPTT)。

### 3. 微觀架構 (`src/micro_circuit.hpp`)
*   **MicroCircuit**：包含 $N$ 個脈衝神經元的遞歸區塊。
*   它具備內部遞歸連接 (`W_rec`) 與殘差路徑，類似於大腦的「皮層柱 (Cortical Column)」，能夠維持局部的時序狀態。

### 4. 宏觀架構 (`src/cortical_hub.hpp`, `src/brain_block.hpp`)
*   **CorticalHub**：中央路由器。它實作了 **快權重 ($M_t$)**，這是一個透過 Hebb 規則更新的短期記憶矩陣 ($M_{t+1} \leftarrow \lambda M_t + \eta (k \otimes v)$)。
*   **BrainBlock**：高層模組，負責並行運行多個 `MicroCircuit` 並透過 Hub 融合其輸出。

## 🗺️ 開發路線圖 (Roadmap)

專案開發分為 6 個階段（詳見 `docs/brain_inspired_bitnet_design_v1.md`）：

- [x] **階段 0：環境建置**
    - CMake 編譯系統與 SYCL (Intel DPC++) 環境配置。
- [x] **階段 1：基礎算子**
    - 實作 `BitLinear` 與 `SpikingUnit` 並通過梯度驗證。
- [x] **階段 2：微結構原型**
    - 具備遞歸能力的 `MicroCircuit` 與基礎快權重 `CorticalHub`。
- [x] **階段 3：BrainBlock 整合**
    - 協調並行電路運作與輸出池化 (Pooling)。
- [ ] **階段 4：進階快權重與解碼**
    - 記憶注意力機制的完整整合，以及防崩壞 (Anti-collapse) 解碼策略。
- [ ] **階段 5：元控制器 (ACC)**
    - 實作衝突監測系統，以動態調整推理深度。
- [ ] **階段 6：持續學習**
    - 情節緩衝區 (Episodic buffers) 與基於 Adapter 的可塑性。

## 🚀 快速開始

### 環境需求
*   **C++ 編譯器**：需支援 C++17 標準。
*   **SYCL SDK**：推薦使用 Intel oneAPI Base Toolkit (DPC++) 以獲得最廣泛的硬體支援 (CPU, GPU, FPGA)。
*   **CMake**：3.10 或更高版本。

### 編譯指南

```bash
# 1. 複製專案庫
git clone https://github.com/your-username/bibna.git
cd bibna

# 2. 建立編譯目錄
mkdir build && cd build

# 3. 配置專案 (請確保已載入 Intel oneAPI 環境變數)
cmake ..

# 4. 編譯
make -j$(nproc)
```

### 執行測試

編譯出的執行檔 `bibna_main` 包含了一套驗證測試，涵蓋單元測試到小規模生成的演示。

```bash
./bibna_main
```

## 📂 專案結構

```
bibna/
├── CMakeLists.txt          # 編譯配置
├── docs/                   # 設計文檔
│   └── brain_inspired...   # 本架構的設計「聖經」
├── main.cpp                # 程式入口點與測試套件
└── src/                    # 原始碼
    ├── acc.hpp             # 衝突監測 (部分實作)
    ├── bitlinear.hpp       # 1.58-bit 線性層
    ├── brain_block.hpp     # 並行電路管理器
    ├── cortical_hub.hpp    # 快權重與注意力
    ├── kernels.hpp         # SYCL 核心 (底層實作)
    ├── micro_circuit.hpp   # 遞歸單元
    └── spiking_unit.hpp    # 神經元模型
```

## 🤝 貢獻指南

歡迎提交貢獻！在提交 Pull Request 之前，請務必詳細閱讀 `docs/` 目錄下的設計文檔。由於這是一個實驗性架構，我們重視 **清晰且有文檔說明的程式碼**，更甚於單純的功能堆疊。

## 📜 授權條款

本專案採用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 授權。詳細內容請參閱 [LICENSE](LICENSE) 檔案。
