import sys
import os
import sys

# --- Path Setup ---
# Add the project root to the Python path to allow imports from other directories like 'util'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 匯入所需函式庫
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, roc_auc_score
from datetime import datetime
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import scipy.sparse
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score

# --- 輔助函數：生物特徵工程 ---
def calculate_aac(sequences):
    """計算胺基酸組成 (Amino Acid Composition)"""
    # 20種標準胺基酸
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_counts = []
    
    for seq in sequences:
        seq = str(seq).upper()
        counts = {aa: 0 for aa in amino_acids}
        length = len(seq)
        if length == 0:
            # Handle empty sequences
            aac_counts.append(counts)
            continue
            
        for char in seq:
            if char in counts:
                counts[char] += 1
        
        # 正規化為頻率
        for aa in counts:
            counts[aa] /= length
        aac_counts.append(counts)
        
    # 轉換為 DataFrame
    aac_df = pd.DataFrame(aac_counts)
    aac_df.columns = [f"AAC_{col}" for col in aac_df.columns]
    return aac_df

# --- 輔助函數：生物特徵工程 ---
def calculate_dpc(sequences):
    """計算雙肽組成 (Dipeptide Composition)"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    dpc_counts = []

    for seq in sequences:
        seq = str(seq).upper()
        counts = {dp: 0 for dp in dipeptides}
        length = len(seq)
        if length < 2:
            dpc_counts.append(counts)
            continue
        
        for i in range(length - 1):
            dp = seq[i : i + 2]
            if dp in counts:
                counts[dp] += 1
        
        # 正規化為頻率
        for dp in counts:
            counts[dp] /= (length - 1)
        dpc_counts.append(counts)
    
    dpc_df = pd.DataFrame(dpc_counts)
    dpc_df.columns = [f"DPC_{col}" for col in dpc_df.columns]
    return dpc_df

# --- 輔助函數：生物特徵工程 ---
def calculate_physchem_properties(sequences):
    """計算序列的平均理化性質"""
    # 資料來源: Kyte & Doolittle hydrophobicity, and other standard sources
    physchem_props = {
        'A': {'hydro': 1.8, 'mw': 89.09, 'pi': 6.00}, 'R': {'hydro': -4.5, 'mw': 174.20, 'pi': 10.76},
        'N': {'hydro': -3.5, 'mw': 132.12, 'pi': 5.41}, 'D': {'hydro': -3.5, 'mw': 133.10, 'pi': 2.77},
        'C': {'hydro': 2.5, 'mw': 121.16, 'pi': 5.07}, 'Q': {'hydro': -3.5, 'mw': 146.15, 'pi': 5.65},
        'E': {'hydro': -3.5, 'mw': 147.13, 'pi': 3.22}, 'G': {'hydro': -0.4, 'mw': 75.07, 'pi': 5.97},
        'H': {'hydro': -3.2, 'mw': 155.16, 'pi': 7.59}, 'I': {'hydro': 4.5, 'mw': 131.17, 'pi': 6.02},
        'L': {'hydro': 3.8, 'mw': 131.17, 'pi': 5.98}, 'K': {'hydro': -3.9, 'mw': 146.19, 'pi': 9.74},
        'M': {'hydro': 1.9, 'mw': 149.21, 'pi': 5.74}, 'F': {'hydro': 2.8, 'mw': 165.19, 'pi': 5.48},
        'P': {'hydro': -1.6, 'mw': 115.13, 'pi': 6.30}, 'S': {'hydro': -0.8, 'mw': 105.09, 'pi': 5.68},
        'T': {'hydro': -0.7, 'mw': 119.12, 'pi': 5.60}, 'W': {'hydro': -0.9, 'mw': 204.23, 'pi': 5.89},
        'Y': {'hydro': -1.3, 'mw': 181.19, 'pi': 5.66}, 'V': {'hydro': 4.2, 'mw': 117.15, 'pi': 5.96}
    }
    prop_names = ['hydro', 'mw', 'pi']
    
    results = []
    for seq in sequences:
        seq = str(seq).upper()
        length = len(seq)
        avg_props = {prop: 0.0 for prop in prop_names}
        
        if length == 0:
            results.append(avg_props)
            continue
        
        valid_chars = 0
        for char in seq:
            if char in physchem_props:
                valid_chars += 1
                for prop in prop_names:
                    avg_props[prop] += physchem_props[char][prop]
        
        if valid_chars > 0:
            for prop in prop_names:
                avg_props[prop] /= valid_chars
        
        results.append(avg_props)
        
    pchem_df = pd.DataFrame(results)
    pchem_df.columns = [f"PChem_Avg_{col}" for col in pchem_df.columns]
    return pchem_df

# --- 0. 設定參數 ---
task_name = 'classify_multilabel'
classify_data_filename = 'train_t1122classify.xlsx - Sheet1.csv'



# 設定資料路徑

data_path = os.path.join(project_root, 'data', 'classification', classify_data_filename)

# 檢查檔案是否存在
if not os.path.exists(data_path):
    print(f"錯誤：在指定的路徑中找不到資料檔案。請檢查 '{data_path}' 是否存在。")
    exit()

# 使用 pandas 讀取資料集
print(f"正在從 {data_path} 讀取資料...")
full_df = pd.read_csv(data_path)

# 為避免因欄位名稱大小寫不一致 (如 'ID' vs 'id') 導致的錯誤，
# 將所有欄位名稱統一轉換為大寫，以進行標準化。
full_df.columns = [col.upper() for col in full_df.columns]

# --- 2. 特徵與目標定義 ---

# 定義不會用於模型訓練的欄位
ID_COL = 'ID'
TARGET_COLS = ['S-GLUTATHIONYLATION', 'S-NITROSYLATION', 'S-PALMITOYLATION'] # 多標籤目標欄位

# ID 欄位也可能不存在，進行保護
if ID_COL not in full_df.columns:
    if 'ID' in full_df.columns:
        ID_COL = 'ID'
    else:
        print("警告：在資料集中找不到 'ID' 欄位。")

# 定義特徵欄位 (features)
features = [col for col in full_df.columns if col not in [ID_COL] + TARGET_COLS]

# --- 3. 特徵工程 (Feature Engineering) ---

# a. 計算 AAC (胺基酸組成) 特徵
if 'SEQUENCE' in full_df.columns:
    print("正在計算胺基酸組成 (AAC) 特徵...")
    aac_df = calculate_aac(full_df['SEQUENCE'])
    full_df = pd.concat([full_df.reset_index(drop=True), aac_df.reset_index(drop=True)], axis=1)
    features.extend(aac_df.columns)
    print(f"已生成 {aac_df.shape[1]} 個 AAC 特徵。")

# b. 計算 DPC (雙肽組成) 特徵
if 'SEQUENCE' in full_df.columns:
    print("正在計算雙肽組成 (DPC) 特徵...")
    dpc_df = calculate_dpc(full_df['SEQUENCE'])
    full_df = pd.concat([full_df.reset_index(drop=True), dpc_df.reset_index(drop=True)], axis=1)
    features.extend(dpc_df.columns)
    print(f"已生成 {dpc_df.shape[1]} 個 DPC 特徵。")

# c. 計算理化性質特徵
if 'SEQUENCE' in full_df.columns:
    print("正在計算平均理化性質特徵...")
    pchem_df = calculate_physchem_properties(full_df['SEQUENCE'])
    full_df = pd.concat([full_df.reset_index(drop=True), pchem_df.reset_index(drop=True)], axis=1)
    features.extend(pchem_df.columns)
    print(f"已生成 {pchem_df.shape[1]} 個理化性質特徵。")

# d. 準備密集特徵 (Dense Features)
numerical_feature_cols = [col for col in features if col != 'SEQUENCE']
imputation_values = full_df[numerical_feature_cols].median()
full_df[numerical_feature_cols] = full_df[numerical_feature_cols].fillna(imputation_values)
dense_features = full_df[numerical_feature_cols].values

# e. 準備稀疏特徵 (Sparse Features) 並合併
if 'SEQUENCE' in full_df.columns:
    print("正在使用 CountVectorizer 生成 k-mer 特徵...")
    sequences = full_df['SEQUENCE'].astype(str)
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3), max_features=10000)
    sparse_kmer_features = vectorizer.fit_transform(sequences)
    print(f"已生成 {sparse_kmer_features.shape[1]} 個 k-mer 特徵 (稀疏格式)。")

    print("正在合併所有特徵...")
    X_combined = hstack([scipy.sparse.csr_matrix(dense_features), sparse_kmer_features]).tocsr()
    
    kmer_names = vectorizer.get_feature_names_out()
    final_feature_names = numerical_feature_cols + list(kmer_names)
    print("特徵合併完成。")
else:
    X_combined = dense_features
    final_feature_names = numerical_feature_cols

# --- 4. 模型訓練 (使用獨立驗證集) ---

# 獲取訓練集和測試集的索引
train_indices, test_indices = train_test_split(
    full_df.index,
    test_size=0.2,
    random_state=42
)

# 使用索引分割特徵矩陣和目標 DataFrame
X_train_full = X_combined[train_indices]
X_test = X_combined[test_indices]
train_df = full_df.loc[train_indices]
test_df = full_df.loc[test_indices]


from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score

# 為每個目標變數訓練一個獨立的模型
trained_models = {}
val_accuracies = {}
val_aucs = {} # 用於儲存每個目標的 AUC 分數
optimal_thresholds = {} # 用於儲存每個目標的最佳閾值
test_predictions_df = pd.DataFrame({ID_COL: test_df[ID_COL]}) # 儲存測試集 ID 和預測結果

for target_col in TARGET_COLS:
    print(f"\n--- 訓練模型：{target_col} ---")
    
    # 定義此目標的 X 和 y (使用完整的訓練資料)
    X = X_train_full
    y = train_df[target_col]
    
    # 將"完整訓練集"分割為"模型訓練集"與"驗證集"
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.25, # 從80%的訓練資料中切出25%作為驗證
        random_state=42,
        stratify=y # 維持原始分佈
    )

    # --- 2. 使用 RandomizedSearchCV 進行超參數搜索 ---
    print("正在設定超參數搜索...")

    # 定義基礎 XGBoost 模型
    base_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device='cuda'
    )

    # 使用 BalancedBaggingClassifier 包裝基礎模型
    # 注意：這裡 n_estimators 是 bagging 的集成數量，而 base_xgb 裡的 n_estimators 會在搜索範圍中定義
    bagging_model = BalancedBaggingClassifier(
        estimator=base_xgb,
        n_estimators=10,
        sampling_strategy='auto',
        replacement=False,
        random_state=42,
        n_jobs=-1
    )

    # 定義要搜索的超參數範圍
    # 格式為 'estimator__<param_name>' 來指定基礎模型的參數
    param_dist = {
        'estimator__learning_rate': [0.01, 0.02, 0.05, 0.1],
        'estimator__max_depth': [5, 7, 9, 11],
        'estimator__n_estimators': [300, 500, 800, 1000],
        'estimator__gamma': [0.1, 0.5, 1, 1.5],
        'estimator__subsample': [0.7, 0.8, 0.9],
        'estimator__colsample_bytree': [0.7, 0.8, 0.9],
        'estimator__min_child_weight': [1, 5, 10]
    }

    # 設定 RandomizedSearchCV
    # n_iter: 隨機抽樣的參數組合數量
    # cv: 交叉驗證的折數
    random_search = RandomizedSearchCV(
        bagging_model,
        param_distributions=param_dist,
        n_iter=15,  # 嘗試 15 種組合，可以根據時間預算調整
        cv=3,
        scoring='roc_auc',
        n_jobs=1,  # n_jobs for RandomizedSearchCV itself, inner n_jobs are -1
        verbose=3,
        random_state=42
    )

    print(f"開始為目標 {target_col} 進行超參數搜索...")
    # 直接在原始(不平衡)訓練集上訓練，RandomizedSearchCV 會處理交叉驗證
    # BalancedBaggingClassifier 會在每個 fold 內部進行平衡採樣
    random_search.fit(X_train, y_train)
    
    print("\n搜索完成！")
    print(f"找到的最佳參數組合: {random_search.best_params_}")
    print(f"對應的最佳 ROC AUC 分數 (交叉驗證): {random_search.best_score_:.4f}")

    # 將最佳模型賦值給 model 變數以供後續使用
    model = random_search.best_estimator_
    
    print(f"模型 {target_col} 訓練完成！")

    # 在(原始不平衡)驗證集上評估模型
    val_predictions = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    val_accuracies[target_col] = val_accuracy
    val_aucs[target_col] = val_auc
    
    print(f"\n--- {target_col} 驗證結果 (預設 0.5 閾值) ---")
    print(f"Accuracy: {val_accuracy:.6f}")
    print(f"ROC AUC: {val_auc:.6f}")
    print("Classification Report:")
    print(classification_report(y_val, val_predictions, target_names=['Class 0', 'Class 1']))
    print("---------------------------------\n")

    # --- 3. 閾值微調 (Threshold Tuning) ---
    print("正在尋找最佳預測閾值以最大化 F1-Score...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_val, val_pred_proba >= t) for t in thresholds]
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    optimal_thresholds[target_col] = best_threshold

    print(f"找到最佳 F1-Score: {best_f1:.6f} (在閾值 = {best_threshold:.2f} 時)")
    
    # 使用最佳閾值產生最終驗證報告
    val_predictions_tuned = (val_pred_proba >= best_threshold).astype(int)
    print(f"\n--- {target_col} 驗證結果 (使用最佳 F1 閾值) ---")
    print(classification_report(y_val, val_predictions_tuned, target_names=['Class 0', 'Class 1']))
    print("---------------------------------\n")

    trained_models[target_col] = model

for target_col in TARGET_COLS:
    # 對於從 full_df 分割出來的 test_df 進行預測 (使用為該目標找到的最佳閾值)
    test_pred_proba = trained_models[target_col].predict_proba(X_test)[:, 1]
    test_predictions_df[target_col] = (test_pred_proba >= optimal_thresholds[target_col]).astype(int)

# 計算並打印所有模型的平均分數
average_accuracy = np.mean(list(val_accuracies.values()))
average_auc = np.mean(list(val_aucs.values()))
print(f"\n所有目標變數的平均 Accuracy: {average_accuracy:.6f}")
print(f"所有目標變數的平均 ROC AUC: {average_auc:.6f}")
val_score = average_auc # 將 val_score 設為平均 ROC AUC





# 建立用於提交的 DataFrame
submission_df = test_predictions_df.copy() # test_predictions_df 已經包含了 ID 和所有目標的預測

# 將欄位名稱轉換為小寫以符合提交格式
submission_df.columns = [col.lower() for col in submission_df.columns]

# The 'id' column contains alphanumeric strings, so we do not convert it to int.
# submission_df['id'] = submission_df['id'].astype(int)
for col in [c.lower() for c in TARGET_COLS]:
    submission_df[col] = submission_df[col].astype(int)

# --- 6. 產生有版本標記的提交檔案 ---
output_dir = os.path.join(project_root, "run", "results", task_name)
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
method_name = "XGBoost_MultiLabel"
submission_filename = f"submission_{method_name}_{timestamp}.csv"
output_path = os.path.join(output_dir, submission_filename)

submission_df.to_csv(output_path, index=False)

print(f"提交檔案 {output_path} 已建立。")

# --- 7. 記錄分數 ---
from util.scoreboard_manager import update_scoreboard

scoreboard_file = os.path.join(project_root, "run", "scoreboard_classify.csv")
score_label = "Average_Accuracy_Score" # 對於多標籤分類任務，我們記錄平均 Accuracy 分數

new_score_entry = pd.DataFrame([{
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Method": method_name,
    "Task": task_name,
    score_label: val_score, # val_score 已經是平均 Accuracy
    "OutputFile": os.path.relpath(output_path, start=project_root)
}])

update_scoreboard(scoreboard_file, new_score_entry)

print(f"分數已更新至 {scoreboard_file}")
