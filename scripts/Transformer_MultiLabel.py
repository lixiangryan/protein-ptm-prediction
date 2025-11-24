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

task_name = 'classify_multilabel_transformer'
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
    TRAINING_PARAMS = {'epochs': 100, 'batch_size': {'transformer': 128}} # Fallback
except Exception as e:
    print(f"讀取 config.yml 時發生錯誤: {e}。將使用預設參數。")
    TRAINING_PARAMS = {'epochs': 100, 'batch_size': {'transformer': 128}} # Fallback

data_path = os.path.join(project_root, 'data', 'classification', classify_data_filename)

if not os.path.exists(data_path):
    print(f"錯誤：在指定的路徑中找不到資料檔案。請檢查 '{data_path}' 是否存在。")
    sys.exit()

# --- 1. 資料載入 ---
print(f"正在從 {data_path} 讀取資料...")
full_df = pd.read_csv(data_path)
full_df.columns = [col.upper() for col in full_df.columns]

# --- 2. 特徵與目標定義 ---
ID_COL = 'ID'
TARGET_COLS = ['S-GLUTATHIONYLATION', 'S-NITROSYLATION', 'S-PALMITOYLATION']

if 'SEQUENCE' not in full_df.columns:
    print("錯誤：資料集中找不到 'SEQUENCE' 欄位。")
    sys.exit()

# --- 3. 序列預處理 (Integer Encoding + Padding) ---
print("正在預處理序列數據...")

# 定義胺基酸字母表
amino_acids = 'ACDEFGHIKLMNPQRSTVWYX' # X for unknown/padding
char_to_int = {c: i for i, c in enumerate(amino_acids)}
vocab_size = len(amino_acids)

# 將序列轉換為整數序列
def sequence_to_int(sequence):
    return [char_to_int.get(char.upper(), char_to_int['X']) for char in str(sequence)]

full_df['SEQUENCE_INT'] = full_df['SEQUENCE'].apply(sequence_to_int)

# 序列填充 (Padding)
max_sequence_len = full_df['SEQUENCE_INT'].apply(len).max()
print(f"最長序列長度: {max_sequence_len}")

X_sequences_padded = keras.preprocessing.sequence.pad_sequences(
    full_df['SEQUENCE_INT'], maxlen=max_sequence_len, padding='post', value=char_to_int['X']
)
print(f"整數編碼與填充後的序列數據形狀: {X_sequences_padded.shape}")

# --- 4. 目標變數準備 ---
y_targets = full_df[TARGET_COLS].values

# --- 5. 訓練集與測試集分割 ---
train_indices, test_indices = train_test_split(
    np.arange(len(full_df)), test_size=0.2, random_state=42
)

X_train_full, X_test = X_sequences_padded[train_indices], X_sequences_padded[test_indices]
y_train_full, y_test = y_targets[train_indices], y_targets[test_indices]
full_df_train, full_df_test = full_df.iloc[train_indices], full_df.iloc[test_indices]

# --- 6. 模型建構 (Transformer) ---

# 6.1. Token & Position Embedding Layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# 6.2. Transformer Encoder Block Layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 6.3. 組合模型
embed_dim = 64  # Embedding a cada token en un vector de 64 dimensiones
num_heads = 4  # Número de cabezales de atención
ff_dim = 64  # Dimensión de la capa oculta en la red feed-forward

inputs = layers.Input(shape=(max_sequence_len,))
embedding_layer = TokenAndPositionEmbedding(max_sequence_len, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x) # Similar a la CNN, reducimos la dimensionalidad
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(TARGET_COLS), activation="sigmoid")(x)

transformer_model = keras.Model(inputs=inputs, outputs=outputs)
transformer_model.summary()

# --- 7. 模型編譯與訓練 ---
print("\n開始訓練 Transformer 模型...")

transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.AUC(name='auc', multi_label=True),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
]

# 分割訓練集為訓練和驗證集
X_train_tf, X_val_tf, y_train_tf, y_val_tf = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42,
    stratify=np.argmax(y_train_full, axis=1) if len(TARGET_COLS) > 1 else y_train_full
    )
    
    history = transformer_model.fit(
        X_train_tf, y_train_tf,
        epochs=TRAINING_PARAMS.get('epochs', 100),
        batch_size=TRAINING_PARAMS.get('batch_size', {}).get('transformer', 128),
        validation_data=(X_val_tf, y_val_tf),
        callbacks=callbacks,
        verbose=1
    )
    # --- 8. 模型評估與閾值微調 ---
print("\n正在評估模型並尋找最佳閾值...")
y_pred_proba_val = transformer_model.predict(X_val_tf)
optimal_thresholds = {}
y_pred_final_test = np.zeros_like(y_test)

for i, target_col in enumerate(TARGET_COLS):
    print(f"\n--- 評估目標：{target_col} ---")
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_val_tf[:, i], (y_pred_proba_val[:, i] >= t)) for t in thresholds]
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    optimal_thresholds[target_col] = best_threshold
    
    print(f"最佳 F1-Score: {np.max(f1_scores):.4f} (在閾值 = {best_threshold:.2f} 時)")
    
    val_predictions_tuned = (y_pred_proba_val[:, i] >= best_threshold).astype(int)
    print("Classification Report (驗證集，最佳閾值):")
    print(classification_report(y_val_tf[:, i], val_predictions_tuned, target_names=['Class 0', 'Class 1']))

    y_pred_proba_test = transformer_model.predict(X_test)[:, i]
    y_pred_final_test[:, i] = (y_pred_proba_test >= best_threshold).astype(int)

avg_roc_auc_val = np.mean([roc_auc_score(y_val_tf[:, i], y_pred_proba_val[:, i]) for i in range(len(TARGET_COLS))])
print(f"\n所有目標的平均 ROC AUC (驗證集): {avg_roc_auc_val:.4f}")

avg_accuracy_test = accuracy_score(y_test, y_pred_final_test)
print(f"\n所有目標的平均 Accuracy (測試集，使用最佳閾值): {avg_accuracy_test:.4f}")

# --- 9. 產生提交檔案 ---
output_dir = os.path.join(project_root, "run", "results", task_name)
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
submission_filename = f"transformer_submission_{timestamp}.csv"
output_path = os.path.join(output_dir, submission_filename)

submission_df = full_df_test[[ID_COL]].copy()
submission_df = pd.concat([submission_df.reset_index(drop=True), pd.DataFrame(y_pred_final_test, columns=TARGET_COLS).reset_index(drop=True)], axis=1)

submission_df.columns = [col.lower() for col in submission_df.columns]
for col in [c.lower() for c in TARGET_COLS]:
    submission_df[col] = submission_df[col].astype(int)

submission_df.to_csv(output_path, index=False)
print(f"提交檔案 {output_path} 已建立。")

# --- 10. 記錄分數 ---
from util.scoreboard_manager import update_scoreboard

scoreboard_file = os.path.join(project_root, "run", "scoreboard_transformer.csv")
score_label = "Average_ROC_AUC" 

new_score_entry = pd.DataFrame([{
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Method": "Transformer",
    "Task": task_name,
    score_label: avg_roc_auc_val,
    "OutputFile": os.path.relpath(output_path, start=project_root)
}])

update_scoreboard(scoreboard_file, new_score_entry)

print(f"分數已更新至 {scoreboard_file}")
