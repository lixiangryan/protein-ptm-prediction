import os
import sys

# --- Path Setup ---
# Add the project root to the Python path to allow imports from other directories like 'util'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml

# --- 0. 設定環境、參數與設定檔 ---
# Add the project root to the Python path to allow imports from other directories like 'util'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

task_name = 'classify_multilabel_cnn'
classify_data_filename = 'train_t1122classify.xlsx - Sheet1.csv'

# 載入設定檔
config_path = os.path.join(project_root, 'config.yml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    TRAINING_PARAMS = config['training']
    print("成功載入 config.yml 設定檔。")
except FileNotFoundError:
    print("錯誤：找不到 config.yml 設定檔。將使用預設參數。")
    TRAINING_PARAMS = {'epochs': 100, 'batch_size': {'cnn': 128}} # Fallback
except Exception as e:
    print(f"讀取 config.yml 時發生錯誤: {e}。將使用預設參數。")
    TRAINING_PARAMS = {'epochs': 100, 'batch_size': {'cnn': 128}} # Fallback


# 設定資料路徑
data_path = os.path.join(project_root, 'data', 'classification', classify_data_filename)

# 檢查檔案是否存在
if not os.path.exists(data_path):
    print(f"錯誤：在指定的路徑中找不到資料檔案。請檢查 '{data_path}' 是否存在。")
    sys.exit()

# --- 1. 資料載入 ---
print(f"正在從 {data_path} 讀取資料...")
full_df = pd.read_csv(data_path)

# 統一欄位名稱為大寫
full_df.columns = [col.upper() for col in full_df.columns]

# --- 2. 特徵與目標定義 ---
ID_COL = 'ID'
TARGET_COLS = ['S-GLUTATHIONYLATION', 'S-NITROSYLATION', 'S-PALMITOYLATION']

# 定義特徵欄位
# 對於 CNN，我們將直接處理 'SEQUENCE' 欄位，其他數值特徵暫時不考慮，以簡化 DL 模型輸入
if 'SEQUENCE' not in full_df.columns:
    print("錯誤：資料集中找不到 'SEQUENCE' 欄位，深度學習模型需要序列數據。")
    sys.exit()

# --- 3. 序列預處理 (One-Hot Encoding) ---
print("正在預處理序列數據...")

# 定義胺基酸字母表
amino_acids = 'ACDEFGHIKLMNPQRSTVWYX' # 考慮可能存在的未知胺基酸 'X'
char_to_int = dict((c, i) for i, c in enumerate(amino_acids))
int_to_char = dict((i, c) for i, c in enumerate(amino_acids))
num_amino_acids = len(amino_acids)

# 將序列轉換為整數序列
def sequence_to_int(sequence):
    return [char_to_int.get(char.upper(), char_to_int['X']) for char in str(sequence) if char.upper() in char_to_int]

full_df['SEQUENCE_INT'] = full_df['SEQUENCE'].apply(sequence_to_int)

# 序列填充 (Padding)
max_sequence_len = full_df['SEQUENCE_INT'].apply(len).max()
print(f"最長序列長度: {max_sequence_len}")

def pad_sequence(sequence_int, max_len):
    if len(sequence_int) >= max_len:
        return sequence_int[:max_len]
    return sequence_int + [char_to_int['X']] * (max_len - len(sequence_int)) # 用 'X' 填充

X_sequences_padded = np.array(full_df['SEQUENCE_INT'].apply(lambda x: pad_sequence(x, max_sequence_len)).tolist())

# One-Hot Encoding
X_one_hot = keras.utils.to_categorical(X_sequences_padded, num_classes=num_amino_acids)
print(f"One-Hot 編碼後的序列數據形狀: {X_one_hot.shape}")

# --- 4. 目標變數準備 ---
y_targets = full_df[TARGET_COLS].values
print(f"目標變數數據形狀: {y_targets.shape}")

# --- 5. 訓練集與測試集分割 ---
# 使用所有數據進行一次分割，確保 ID 與序列匹配
train_indices, test_indices = train_test_split(
    np.arange(len(full_df)), 
    test_size=0.2, 
    random_state=42
)

X_train_full, X_test = X_one_hot[train_indices], X_one_hot[test_indices]
y_train_full, y_test = y_targets[train_indices], y_targets[test_indices]
full_df_train, full_df_test = full_df.iloc[train_indices], full_df.iloc[test_indices]


import keras_tuner as kt

# --- 6. 模型建構 (Hyper-tunable 1D-ResNet) ---
def make_residual_block(input_tensor, filters, kernel_size):
    # ... (This helper function remains the same)
    x = layers.Conv1D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if input_tensor.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(input_tensor)
    else:
        shortcut = input_tensor
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_model(hp):
    """Builds a 1D ResNet model with tunable hyperparameters."""
    input_shape = (max_sequence_len, num_amino_acids)
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(
        filters=hp.Int('init_filters', min_value=32, max_value=128, step=32),
        kernel_size=7, 
        padding='same'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stack of residual blocks
    # Tune the number of filters and blocks
    for i in range(hp.Int('num_resnet_blocks', 1, 3)):
        x = make_residual_block(
            x,
            filters=hp.Int(f'filters_{i}', min_value=64, max_value=256, step=64),
            kernel_size=3
        )
    
    # Final layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    )(x)
    x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    outputs = layers.Dense(len(TARGET_COLS), activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model inside the build function
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return model

# --- 7. KerasTuner 超參數搜索 ---
print("\n設定 KerasTuner 並開始搜索...")

tuner = kt.RandomSearch(
    build_resnet_model,
    objective=kt.Objective('val_auc', direction='max'),
    max_trials=10,  # 嘗試 10 種不同的超參數組合
    executions_per_trial=1, # 每次試驗訓練 1 次
    directory='keras_tuner_dir',
    project_name='cnn_resnet_tuning',
    overwrite=True
)

tuner.search_space_summary()

# 分割訓練集為訓練和驗證集
X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train_full, axis=1) if len(TARGET_COLS) > 1 else y_train_full
)

# 計算樣本權重
print("正在計算樣本權重...")
class_weights_per_target = []
for i in range(y_train_full.shape[1]):
    neg, pos = np.bincount(y_train_full[:, i].astype(int))
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 0
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 0
    class_weights_per_target.append({0: weight_for_0, 1: weight_for_1})

sample_weights = np.array([max([class_weights_per_target[j][label] for j, label in enumerate(sample_label)]) for sample_label in y_train_cnn])
print("樣本權重計算完成。")

# --- 建立高效的 tf.data 管線 ---
print("建立 tf.data 高效能管線...")
batch_size_cnn = TRAINING_PARAMS.get('batch_size', {}).get('cnn', 128)

# 將樣本權重與特徵和目標打包
# Note: KerasTuner's search method doesn't directly accept sample_weight with a tf.data.Dataset.
# The recommended way is to make the dataset yield a tuple of (inputs, targets, sample_weights).
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_cnn, y_train_cnn, sample_weights))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val_cnn))

# 建立管線：打亂 -> 分批 -> 預取
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size_cnn).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size_cnn).prefetch(tf.data.AUTOTUNE)
print("tf.data 管線建立完成。")

# 執行搜索
tuner.search(
    train_dataset,
    epochs=20, # 在搜索階段使用較少的 epochs
    validation_data=val_dataset,
    callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
)

print("\n超參數搜索完成！")
# 獲取最佳模型
cnn_model = tuner.get_best_models(num_models=1)[0]
# 顯示最佳參數
tuner.results_summary()

# (可選) 用最佳參數重新訓練更長的時間
print("\n使用最佳參數重新訓練最終模型...")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
cnn_model = build_resnet_model(best_hps) # Re-build the model with the best HPs

history = cnn_model.fit(
    train_dataset, # Use the optimized dataset
    epochs=TRAINING_PARAMS.get('epochs', 100),
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=1
)


# --- 8. 模型評估與閾值微調 ---
print("\n正在評估模型並尋找最佳閾值...")
y_pred_proba_val = cnn_model.predict(X_val_cnn)
optimal_thresholds = {}
y_pred_final = np.zeros_like(y_test)

for i, target_col in enumerate(TARGET_COLS):
    print(f"\n--- 評估目標：{target_col} ---")
    
    # 針對驗證集尋找最佳閾值
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_val_cnn[:, i], (y_pred_proba_val[:, i] >= t)) for t in thresholds]
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    optimal_thresholds[target_col] = best_threshold
    
    print(f"最佳 F1-Score: {np.max(f1_scores):.4f} (在閾值 = {best_threshold:.2f} 時)")
    
    # 使用最佳閾值在驗證集上報告
    val_predictions_tuned = (y_pred_proba_val[:, i] >= best_threshold).astype(int)
    print("Classification Report (驗證集，最佳閾值):")
    print(classification_report(y_val_cnn[:, i], val_predictions_tuned, target_names=['Class 0', 'Class 1']))

    # 對於測試集，使用找到的最佳閾值進行預測
    y_pred_proba_test = cnn_model.predict(X_test)[:, i]
    y_pred_final[:, i] = (y_pred_proba_test >= best_threshold).astype(int)

# 計算所有目標的平均 ROC AUC (在驗證集上)
avg_roc_auc_val = np.mean([roc_auc_score(y_val_cnn[:, i], y_pred_proba_val[:, i]) for i in range(len(TARGET_COLS))])
print(f"\n所有目標的平均 ROC AUC (驗證集): {avg_roc_auc_val:.4f}")

# 計算測試集上的總體平均 Accuracy
avg_accuracy_test = accuracy_score(y_test, y_pred_final)
print(f"\n所有目標的平均 Accuracy (測試集，使用最佳閾值): {avg_accuracy_test:.4f}")

# --- 9. 產生提交檔案 ---
output_dir = os.path.join(project_root, "run", "results", task_name)
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
submission_filename = f"cnn_submission_{timestamp}.csv"
output_path = os.path.join(output_dir, submission_filename)

submission_df = full_df_test[[ID_COL]].copy()
submission_df = pd.concat([submission_df.reset_index(drop=True), pd.DataFrame(y_pred_final, columns=TARGET_COLS).reset_index(drop=True)], axis=1)

submission_df.columns = [col.lower() for col in submission_df.columns]
for col in [c.lower() for c in TARGET_COLS]:
    submission_df[col] = submission_df[col].astype(int)

submission_df.to_csv(output_path, index=False)
print(f"提交檔案 {output_path} 已建立。")

# --- 10. 記錄分數 (簡化處理，僅記錄平均 ROC AUC) ---
from util.scoreboard_manager import update_scoreboard

scoreboard_file = os.path.join(project_root, "run", "scoreboard_cnn.csv")
score_label = "Average_ROC_AUC" 

new_score_entry = pd.DataFrame([{
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Method": "1D-CNN",
    "Task": task_name,
    score_label: avg_roc_auc_val,
    "OutputFile": os.path.relpath(output_path, start=project_root)
}])

update_scoreboard(scoreboard_file, new_score_entry)

print(f"分數已更新至 {scoreboard_file}")
