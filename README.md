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

本專案已透過 Docker 容器化，以確保環境一致性和易於部署。請確保您的系統已安裝 Docker Desktop，且若需使用 GPU，NVIDIA GPU 驅動程式已正確安裝。

### 3.1. 建置 Docker 映像 (Build Docker Image)

1.  **準備工作**：
    *   確保您的終端機已進入專案的根目錄 (`D:\學校相關\政大\課碩一上\Machine Learning\pochen`)。
    *   請確認 `requirements.txt` 檔案存在且內容正確。
    *   若您之前有 `requirements_tf.txt` 檔案，為避免混淆，請手動將其刪除。

2.  **建置映像**：
    *   執行以下指令來建置 Docker 映像。這會根據 `Dockerfile` 下載所需組件並安裝所有 Python 函式庫。
    ```bash
    docker build -t protein-ptm-predictor .
    ```
    (注意指令末尾的 `.` 代表當前目錄)

### 3.2. 啟動 Docker 容器並執行程式 (Run Docker Container and Execute Scripts)

1.  **啟動容器**：
    *   映像建置完成後，執行以下指令來啟動容器。此指令會將您本機的專案資料夾 (`%cd%` 或 `/path/to/your/project`) 掛載到容器內的 `/app` 目錄，以便您在容器內外同步修改程式碼和存取輸出檔案。
    ```bash
    docker run --gpus all -it --rm -v "%cd%:/app" protein-ptm-predictor
    ```
    *   **指令說明**：
        *   `--gpus all`：賦予容器存取所有可用 GPU 的權限 (若無 GPU，可省略)。
        *   `-it`：以互動模式啟動容器，並分配一個偽終端機 (pseudo-TTY)。
        *   `--rm`：容器停止後自動刪除容器實例 (不影響映像)。
        *   `-v "%cd%:/app"`：將您的當前主機目錄 (`%cd%` 在 Windows 上，或在 Linux/macOS 上替換為 `$(pwd)` 或您的專案絕對路徑) 掛載到容器內的 `/app` 目錄。

2.  **在容器內執行程式**：
    *   成功啟動容器後，您的終端機提示符會變為類似 `root@<container_id>:/app#` 的樣子。此時，您已在 Docker 環境內。
    *   您可以像之前一樣執行專案中的 Python 腳本：
        ```bash
        python XGBoost_MultiLabel.py
        ```
        ```bash
        python CNN_MultiLabel.py
        ```
    *   程式的輸出檔案 (例如 `run/results` 和 `run/scoreboard` 中的 CSV) 將會出現在您的本機專案資料夾中。


## 4. 未來工作 (Future Work)

### 4.1. (已完成) 進階生物特徵工程
此階段任務已成功完成，透過為模型引入生物學領域知識，極大地提升了性能。
- **策略**：除了 k-mer 頻率，我們計算並引入了三組具有明確生物學意義的特徵。
- **第一步：胺基酸組成 (AAC)**。提供了序列整體化學環境的宏觀視角，**使平均 ROC AUC 從 0.633 顯著提升至 0.708**。
- **第二步：雙肽組成 (DPC)**。引入相鄰胺基酸對的頻率資訊後，模型性能再次飛躍，**平均 ROC AUC 從 0.708 提升至 0.734**。
- **第三步：理化性質 (Physicochemical Properties)**。在加入平均疏水性、分子量等特徵後，模型性能達到新高，**平均 ROC AUC 從 0.734 提升至 0.746**。
- **結論**：特徵工程是本次優化中最成功的環節，證明了結合領域知識的重要性。

### 4.2. 探索深度學習模型 (已完成第一步 - 1D-CNN)
此階段旨在利用深度學習強大的特徵自動提取能力，探索性能的上限。

- **已完成：一維卷積神經網路 (1D-CNN)**
    - **策略**：使用 `TensorFlow` 和 `Keras` 構建了一個 1D-CNN 模型，它擅長自動從序列中提取局部模式（Pattern/Motif）。
    - **結果**：CNN 模型在性能上顯著超越了所有優化後的 XGBoost 模型，**平均 ROC AUC 達到 0.772**，證明了此方向的巨大潛力。

- **下一步計畫：Transformer 架構**
    - **策略**：實現一個基於 Transformer 的模型。Transformer 的核心是「自註意力機制 (Self-Attention)」，它能夠捕捉序列中任意兩個胺基酸之間的長距離依賴關係，這是 CNN 的弱點。
    - **目標**：驗證 Transformer 模型是否能透過其對全局依賴性的強大建模能力，進一步提升預測性能，特別是對於那些可能受遠端殘基影響的修飾位點。這代表了當前序列建模領域的最前沿方法。

### 4.3. 潛在的進一步優化空間 (Potential Further Optimization Avenues)

**儘管目前模型性能已顯著提升，但針對 XGBoost, CNN 和 Transformer 各自的特性，仍存在以下優化空間：**

*   **XGBoost 模型 (基於 Tabular/Engineered Features):**
    *   **系統性的超參數搜索 (Systematic Hyperparameter Search):** 運用 `Optuna` 或 `GridSearchCV` 等工具，自動化地尋找最佳的 XGBoost 參數組合。
    *   **進階特徵選擇 (Advanced Feature Selection):** 雖然已引入大量特徵，但可透過演算法篩選出最關鍵、去冗餘的特徵子集，以提高模型效率與泛化能力。
    *   **集成學習 (`BalancedBaggingClassifier`) 的參數優化：** 微調 Bagging 本身的參數，如 `n_estimators` (集成模型數量)。

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
