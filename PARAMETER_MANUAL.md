# 參數設定手冊 (Parameter Configuration Manual)

本文件詳細說明了專案中 `config.yml` 檔案的用途、結構以及如何調整其中的參數，以便使用者能根據自己的硬體設備和實驗需求進行最佳化配置。

---

## 1. `config.yml` 檔案簡介

`config.yml` 是一個 YAML 格式的設定檔，用於集中管理深度學習模型（CNN 和 Transformer）的訓練超參數。透過修改此檔案，您可以輕鬆調整模型的行為，而無需更動任何 Python 程式碼，這極大地提升了專案的靈活性和可維護性。

---

## 2. `config.yml` 結構與參數說明

以下是 `config.yml` 檔案的當前結構和各參數的詳細說明：

```yaml
# Configuration for Model Training

training:
  epochs: 100
  batch_size:
    # Batch size for the CNN model.
    # Note: 128 is a relatively large batch size, optimized for high-end GPUs
    # like NVIDIA RTX 5090 (with 24GB+ VRAM) to maximize GPU utilization.
    # For GPUs with less memory (e.g., 8GB or 12GB), you might need to reduce
    # this value (e.g., 64, 32, or 16) to avoid out-of-memory errors.
    cnn: 128
    # Batch size for the Transformer model.
    # Similar considerations as for CNN apply here.
    transformer: 128
```

### 2.1. `training` 區塊

此區塊包含所有與模型訓練過程相關的參數。

*   **`epochs`**
    *   **類型**：整數
    *   **預設值**：`100`
    *   **說明**：定義了模型在整個訓練數據集上迭代的次數。每次完整的數據集迭代稱為一個 Epoch。
    *   **調整建議**：由於我們啟用了 Early Stopping (如果模型在驗證集上的性能停止提升，就會提前停止訓練)，所以您可以將 `epochs` 設置為一個較大的數值（例如 `100` 或 `200`），讓模型有足夠的機會收斂，而不會因為訓練過久而過擬合。

*   **`batch_size`**
    *   **類型**：整數
    *   **說明**：定義了在每次訓練迭代中，模型會處理的樣本數量。
    *   **子參數**：
        *   **`cnn`**：用於 CNN 模型的批次大小。
        *   **`transformer`**：用於 Transformer 模型的批次大小。
    *   **預設值**：`128`
    *   **調整建議**：
        *   **效能考量**：較大的 `batch_size`（例如 `128`）通常能更好地利用 GPU 的並行計算能力，加快每個 Epoch 的訓練速度。但這也意味著需要更多的 GPU 顯存。
        *   **硬體限制**：目前的預設值 `128` 是為了 NVIDIA RTX 5090 (具有 24GB+ 顯存) 等高階 GPU 進行了優化，以最大化 GPU 利用率。
        *   **低顯存 GPU**：如果您的 GPU 顯存較少（例如 8GB 或 12GB），您可能需要**降低**此值（例如調整為 `64`, `32`, `16`），以避免出現「顯存不足 (Out-of-Memory, OOM)」錯誤。請根據您的硬體情況進行調整，常見的選擇是 2 的冪次方。

---

## 3. 如何修改參數

要調整模型的訓練參數，只需用任何文字編輯器打開 `config.yml` 檔案，修改對應的數值，然後儲存檔案即可。下次運行模型時，腳本會自動讀取新的設定。

---

## 4. (未來擴展) 新增其他參數

`config.yml` 的設計是可擴展的。如果未來需要調整學習率、Dropout 率、模型層數等參數，都可以在此檔案中新增對應的設定，並在腳本中讀取使用，進一步提升模型的配置彈性。
