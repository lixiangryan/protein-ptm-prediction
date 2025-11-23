# 多標籤蛋白質分類模型優化日誌 (Multi-Label Protein Classification Model Optimization Log)

本專案旨在優化一個用於預測蛋白質修飾位點的多標籤分類模型。起始模型在處理高度不平衡的數據集時遇到了嚴重的性能瓶頸，尤其是在 `S-PALMITOYLATION` 這個目標上，其 ROC AUC 分數接近 0.5，意味著模型幾乎沒有預測能力。

本文件記錄了整個問題診斷和模型優化的完整歷程。

## 1. 初始問題：模型失衡與性能瓶頸

### 1.1. 性能指標分析
初始模型雖然在總體準確率 (Accuracy) 上表現尚可（約 95%），但這是一個由數據不平衡導致的假象。深入分析後發現：
- **ROC AUC 分數極低**：對 `S-PALMITOYLATION` 的預測接近 0.515，與隨機猜測無異。
- **召回率 (Recall) 趨近於零**：對正例 (Class 1) 的召回率僅為 0.05，代表模型錯過了 95% 的目標樣本。
- **結論**：模型走了「捷徑」，透過持續預測多數類 (Class 0) 來取得虛高的準確率，但並未學到任何關於少數類的有效特徵。

### 1.2. 技術障礙：記憶體錯誤 (MemoryError)
在嘗試進行更複雜的特徵工程（如 k-mer 分析）時，程式因試圖建立一個超過 5GB 的密集矩陣 (dense matrix) 而崩潰，觸發了 `MemoryError`。

---

## 2. 優化歷程

### 2.1. 基礎建設：修正 MemoryError
為了解決記憶體瓶頸，我們對特徵工程流程進行了根本性的重構。
- **策略**：棄用 Pandas DataFrame 來儲存高維的 k-mer 特徵，改用 `scikit-learn` 的 `CountVectorizer` 來產生**稀疏矩陣 (Sparse Matrix)**。
- **結果**：成功在可控的記憶體使用下，產生了高達數萬維的特徵，為後續優化鋪平了道路。

### 2.2. 第一輪優化：處理數據不平衡
為了解決模型學習偏誤的問題，我們引入了數據平衡策略。
- **策略**：在訓練階段，使用 `imbalanced-learn` 套件中的 **`RandomUnderSampler` (隨機欠採樣)**，將多數類的樣本數量減少到與少數類相同，迫使模型關注少數類的特徵。同時，對 XGBoost 的超參數進行了初步調整。
- **結果**：模型的 Recall 開始顯著提升，證明此方向正確，但整體性能依然不佳。

### 2.3. 第二輪優化：特徵工程擴展
在數據基本平衡後，我們將瓶頸指向了特徵的豐富度。
- **策略**：
    1.  將 k-mer 的分析範圍從單一的 `k=3` 擴展為 `ngram_range=(2, 3)`，以同時捕捉雙肽和三肽模式。
    2.  將 `max_features` 從 5,000 增加到 10,000，納入更多潛在的特徵。
- **結果**：`S-PALMITOYLATION` 的 **Recall 從 0.11 大幅躍升至 0.32**，代價是 Precision 的下降。ROC AUC 也突破了 0.6，證明了特徵豐富化的有效性。

### 2.4. 第三輪優化：預測閾值微調 (Threshold Tuning)
為尋求在 Recall 和 Precision 之間的最佳平衡，我們引入了閾值微調。
- **策略**：在模型訓練後，不使用預設的 0.5 作為分類閾值，而是在驗證集上遍歷 0.01 到 0.99 之間的所有閾值，尋找能使 **F1-Score 最大化**的最佳閾值。
- **結果**：模型的 F1-Score 得到了提升，在 Recall 和 Precision 之間取得了一個更優的平衡點，整體預測結果更具參考價值。

### 2.5. 第四輪優化：集成學習 (Ensemble Learning)
此步驟旨在透過集成方法提升模型的穩定性和性能。
- **策略**：使用 `imbalanced-learn` 的 **`BalancedBaggingClassifier`**，建立一個由 10 個 XGBoost 模型組成的集成「戰隊」。每個模型都在不同的、平衡過的數據子集上訓練。
- **結果**：此方法帶來了顯著的全局性能提升。**平均 ROC AUC 分數從 0.604 提升至 0.633**。這證明了集成學習在處理不平衡數據時的穩定性與優越性。雖然在最困難的目標 `S-PALMITOYLATION` 上 F1-Score 有輕微波動，但整體的辨識能力得到了實質性的增強。此改動被保留作為新的基準模型 (baseline)。
- **備註：關於 GPU 預測性能的警告**：在最終的模型版本中，執行預測時可能會出現一個關於 `mismatched devices` 的 `UserWarning`。這是因為模型在 GPU 上運行，而輸入的稀疏矩陣數據在 CPU 上。我們曾嘗試透過更新 `xgboost` 版本並使用 `xgb.DMatrix(..., device="cuda")` 的語法來修正，但因 API 不相容而導致 `TypeError`。另一個 `cupy` 方案則有引發記憶體錯誤的風險。因此，我們最終決定接受這個不影響結果正確性的性能警告，以確保程式碼的穩定性和通用性。

## 3. 執行方式 (How to Run)

本專案提供兩種執行方式：透過 Conda 本地環境直接執行，或透過 Docker 容器執行。

### 3.1. 方式一：本地端執行 (建議)

這是最簡單直接的執行方式。

1.  **環境設定**：
    *   本專案有兩個 Conda 環境，`MLenv` (Python 3.14) 和 `tf_env` (Python 3.11)。由於 `tf_env` 包含了運行所有模型 (XGBoost, CNN, Transformer) 所需的全部套件，強烈建議使用此環境。
    *   啟用環境：
    ```bash
    conda activate tf_env
    ```

2.  **執行主程式**：
    *   在專案根目錄下，執行 `main.py`：
    ```bash
    python main.py
    ```
    *   程式會顯示一個互動式選單，您可以根據提示選擇要運行的模型 (1: XGBoost, 2: CNN, 3: Transformer)。

### 3.2. 方式二：Docker 執行

此方式可以確保在任何裝有 Docker 的機器上，都能擁有完全一致的執行環境。

1.  **建置映像**：
    *   在專案根目錄下執行以下指令來建置 Docker 映像。
    ```bash
    docker build -t protein-ptm-predictor .
    ```

2.  **啟動容器並執行程式**：
    *   執行以下指令來啟動容器。此指令會將您本機的專案資料夾掛載到容器內，方便同步程式碼和存取輸出檔案。
    ```bash
    docker run --gpus all -it --rm -v "%cd%:/app" protein-ptm-predictor
    ```
    *   **在容器內**：成功啟動後，您的終端機提示符會改變。此時，您已在 Docker 環境內，可以透過 `python main.py` 來啟動選單，或直接執行 `scripts/` 下的任一腳本。

### 3.3. 退出容器 (Exiting the Container)

當您在容器的終端機中完成操作後，只需輸入以下指令並按下 Enter，即可退出並自動關閉/刪除容器：
```bash
exit
```

### 3.4. 參數設定 (Parameter Configuration)

為了方便調整深度學習模型（CNN, Transformer）的訓練參數，專案根目錄下提供了一個 `config.yml` 檔案。

```yaml
# config.yml
training:
  epochs: 100
  batch_size:
    cnn: 128
    transformer: 128
```

您可以直接修改此檔案中的數值，來調整訓練的週期 (`epochs`) 或批次大小 (`batch_size`)，而無需更動任何 Python 程式碼。這對於在不同硬體上進行實驗或微調模型非常方便。

## 4. 未來工作 (Future Work)

### 4.1. (已完成) 進階生物特徵工程
此階段任務已成功完成，透過為模型引入生物學領域知識，極大地提升了性能。
- **策略**：除了 k-mer 頻率，我們計算並引入了三組具有明確生物學意義的特徵。
- **第一步：胺基酸組成 (AAC)**。提供了序列整體化學環境的宏觀視角，**使平均 ROC AUC 從 0.633 顯著提升至 0.708**。
- **第二步：雙肽組成 (DPC)**。引入相鄰胺基酸對的頻率資訊後，模型性能再次飛躍，**平均 ROC AUC 從 0.708 提升至 0.734**。
- **第三步：理化性質 (Physicochemical Properties)**。在加入平均疏水性、分子量等特徵後，模型性能達到新高，**平均 ROC AUC 從 0.734 提升至 0.746**。
- **結論**：特徵工程是本次優化中最成功的環節，證明了結合領域知識的重要性。

### 4.2. 探索深度學習模型 (已完成)
此階段旨在利用深度學習強大的特徵自動提取能力，探索性能的上限。

- **已完成：一維卷積神經網路 (1D-CNN)**
    - **策略**：使用 `TensorFlow` 和 `Keras` 構建了一個 1D-CNN 模型，它擅長自動從序列中提取局部模式（Pattern/Motif）。
    - **結果**：CNN 模型在性能上顯著超越了所有優化後的 XGBoost 模型，**平均 ROC AUC 達到 0.772**，證明了此方向的巨大潛力。

- **已完成：Transformer 架構**
    - **策略**：實現了一個基於 Transformer 的模型，其核心是「自註意力機制 (Self-Attention)」，旨在捕捉序列中胺基酸之間的長距離依賴關係。
    - **結果**：Transformer 模型在性能上再次實現提升，**平均 ROC AUC 達到了 0.780**，超越了 CNN 模型，成為專案中性能最強的模型。這證實了 Transformer 架構在蛋白質序列分析中的前沿地位和優越性。

### 4.3. 潛在的進一步優化空間 (Potential Further Optimization Avenues)

**儘管目前模型性能已顯著提升，但針對 XGBoost, CNN 和 Transformer 各自的特性，仍存在以下優化空間：**

*   **XGBoost 模型 (基於 Tabular/Engineered Features):**
    *   **系統性的超參數搜索 (Systematic Hyperparameter Search):** **(已完成)** 在 `XGBoost_MultiLabel.py` 中導入了 `RandomizedSearchCV`，以自動化、系統性地搜索最佳的超參數組合。
    *   **進階特徵選擇 (Advanced Feature Selection):** **(已作為可選模式實現)** 腳本現在支援 `--mode select` 參數，可在訓練前先運行一個初步模型來評估特徵重要性，並只用最重要的 N 個特徵（預設 2000）來訓練最終模型。此功能可透過 `main.py` 的選單來啟用。
    *   **集成學習 (`BalancedBaggingClassifier`) 的參數優化：** 在超參數搜索中，也可以包含對 Bagging 本身參數（如 `n_estimators`）的優化。

*   **CNN 模型 (基於 Sequence - 自動學習局部模式):**
    *   **樣本加權 (Sample Weighting):** 在訓練時對少數類樣本賦予更高權重，進一步解決類別不平衡問題。
    *   **架構超參數調優:** 調整卷積核數量、大小、層數，池化策略，以及全連接層的設計。可使用自動化工具如 `KerasTuner`。
    *   **更複雜的 CNN 結構:** 探索多分支 CNN (Inception-like)、殘差連接 (ResNet-like) 等更複雜的架構。

*   **Transformer 模型 (基於 Sequence - 自動學習全局關聯):**
    *   **超參數調優:** 調節注意力頭數量 (num_heads)、前饋網路維度 (ff_dim)、編碼器層數等。
    *   **預訓練策略 (Pre-training):** 利用大量無標籤蛋白質序列數據進行預訓練（如 Masked Language Modeling），再對特定任務進行微調，這是 Transformer 在自然語言處理領域取得成功的關鍵。
    *   **更複雜的 Transformer 結構:** 探索更深層次的 Transformer 編碼器堆疊。

*   **異質模型堆疊 (Heterogeneous Model Stacking):**
    *   將 XGBoost (從結構化特徵學習) 和 CNN/Transformer (從原始序列學習) 的預測結果進行集成，透過訓練一個「元模型」來學習如何最佳地組合不同模型的優勢，有望實現最終的性能突破。

## 5. 關於模型可解釋性 (On Model Interpretability)

在本專案中，我們使用了 XGBoost、CNN 和 Transformer 三種不同的模型。它們在「可解釋性」上存在明顯的層次差異。

### 5.1. XGBoost (可解釋性：高)

*   **為何可解釋？**
    *   **特徵重要性 (Feature Importance)**：XGBoost 能直接計算並排列出所有我們手動設計的特徵（如 AAC, DPC, k-mers 等）對預測的貢獻度。這讓我們能直觀地理解模型是基於哪些「規則」來做判斷的。
*   **如何解釋？**
    *   透過分析排名前列的特徵，我們可以得出宏觀結論，例如「DPC 特徵普遍比 AAC 重要」，或「某個特定理化性質是關鍵驅動因素」。

### 5.2. CNN (可解釋性：中等)

*   **為何是黑盒子？**
    *   CNN 自己學習特徵，其內部卷積層的權重對人類來說只是一堆無意義的數字。
*   **如何「窺探」黑盒子？**
    *   **濾波器視覺化 (Filter Visualization)**：分析卷積核對哪些輸入模式反應最強烈，從而推斷出模型自動學習到的「序列基序 (Motif)」。
    *   **顯著圖 (Saliency Map)**：反向計算輸入序列中每個胺基酸對單次預測結果的貢獻度，並用熱力圖將其視覺化，以了解模型在做決策時「關注」了哪些位置。

### 5.3. Transformer (可解釋性：中等)

*   **為何是黑盒子？**
    *   與 CNN 類似，它是一個非常深且複雜的神經網路。
*   **如何「窺探」黑盒子？**
    *   **注意力圖 (Attention Map)**：這是 Transformer 獨有的可解釋性工具。我們可以將「自註意力機制」所計算的胺基酸之間的「關注分數」視覺化成熱力圖。這能讓我們直觀地理解模型學到的**長距離和短距離依賴關係**，這是 CNN 和 XGBoost 都無法直接提供的洞見。

### 總結

| 模型 | 可解釋性 | 主要方法 | 能告訴我們什麼？ |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **高** | 特徵重要性 (Feature Importance) | 「哪些**我們設計的特徵**最重要？」 |
| **CNN** | **中等** | 濾波器視覺化、顯著圖 | 「模型自己學會了哪些**局部序列模式(Motif)**？」 |
| **Transformer**| **中等** | 注意力圖 (Attention Map) | 「模型認為序列中**哪些胺基酸之間存在重要關聯**（不論遠近）？」 |

