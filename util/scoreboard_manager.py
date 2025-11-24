import pandas as pd
import os
from io import StringIO

# Define project_root at the module level for use in all functions
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def update_scoreboard(scoreboard_file, new_score_entry):
    """
    Updates the scoreboard, keeping the best scores for specified tasks at the top.
    """
    history_df = pd.DataFrame()
    
    # --- 1. Read existing history ---
    if os.path.exists(scoreboard_file):
        try:
            with open(scoreboard_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            blank_line_indices = [i for i, line in enumerate(lines) if line.strip() == '']
            if blank_line_indices:
                history_start_line = blank_line_indices[0] + 1
                if len(lines) > history_start_line:
                    history_csv = StringIO("".join(lines[history_start_line:]))
                    history_df = pd.read_csv(history_csv)
            else: # If no blank line, the whole file is history
                history_csv = StringIO("".join(lines))
                history_df = pd.read_csv(history_csv)

        except (pd.errors.EmptyDataError, FileNotFoundError, IndexError):
            history_df = pd.DataFrame()

    # --- 2. Append new entry and sort ---
    history_df = pd.concat([history_df, new_score_entry], ignore_index=True)
    history_df = history_df.drop_duplicates(subset=['Timestamp', 'Method', 'Task'], keep='last')
    
    # Convert to datetime for sorting, then back to string
    history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
    history_df = history_df.sort_values(by="Timestamp", ascending=False)
    history_df['Timestamp'] = history_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- 3. Prepare best scores DataFrame ---
    score_col = None
    if 'RMSE_Score' in history_df.columns:
        score_col = 'RMSE_Score'
        sort_ascending = True
    elif 'Accuracy_Score' in history_df.columns:
        score_col = 'Accuracy_Score'
        sort_ascending = False
    elif 'Average_ROC_AUC' in history_df.columns:
        score_col = 'Average_ROC_AUC'
        sort_ascending = False
    
    top_df = pd.DataFrame()
    if score_col:
        valid_scores_df = history_df.dropna(subset=[score_col]).copy()
        valid_scores_df[score_col] = pd.to_numeric(valid_scores_df[score_col])
        
        top_rows = []
        if not valid_scores_df.empty:
            top_tasks = valid_scores_df['Task'].unique()
            for task in top_tasks:
                task_df = valid_scores_df[valid_scores_df['Task'] == task]
                task_best_score_row = task_df.sort_values(by=score_col, ascending=sort_ascending).iloc[0]
                top_rows.append(task_best_score_row.to_dict())
        
        if top_rows:
            top_df = pd.DataFrame(top_rows, columns=new_score_entry.columns)

    # --- 4. Rewrite the entire file ---
    try:
        with open(scoreboard_file, 'w', encoding='utf-8', newline='') as f:
            if not top_df.empty:
                top_df.to_csv(f, header=True, index=False, lineterminator='\n')
                f.write('\n')
            history_df.to_csv(f, header=True, index=False, lineterminator='\n')
        return True
            
    except IOError as e:
        print(f"寫入 {scoreboard_file} 時發生錯誤: {e}")
        return False

def clean_history():
    """
    Deletes all but the best result files from run/results, and preserves scoreboard history.
    """
    print("--- Starting History Cleanup ---")
    
    scoreboards_to_process = [
        os.path.join("run", "scoreboard_classify.csv"), # For XGBoost
        os.path.join("run", "scoreboard_cnn.csv"),      # For CNN
        os.path.join("run", "scoreboard_transformer.csv") # For Transformer
    ]
    
    all_best_files = set()

    # --- 1. Find best files to keep ---
    for board_path in scoreboards_to_process:
        scoreboard_file = os.path.join(project_root, board_path)
        if not os.path.exists(scoreboard_file):
            print(f"Skipping {scoreboard_file} (not found).")
            continue

        try:
            with open(scoreboard_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            blank_line_indices = [i for i, line in enumerate(lines) if line.strip() == '']
            if not blank_line_indices:
                print(f"Could not find a best-score summary section in {scoreboard_file}. Cannot determine which files to keep.")
                continue

            top_csv_str = "".join(lines[:blank_line_indices[0]])
            if not top_csv_str.strip():
                print(f"Best-score summary in {scoreboard_file} is empty. Nothing to process.")
                continue

            top_df = pd.read_csv(StringIO(top_csv_str))
            
            if 'OutputFile' in top_df.columns:
                for file_path in top_df['OutputFile']:
                    if isinstance(file_path, str) and file_path != '-':
                        normalized_path = os.path.normpath(os.path.join(project_root, file_path))
                        all_best_files.add(normalized_path)
            
            # This function no longer modifies the scoreboard files.
            print(f"Processed scoreboard {scoreboard_file} to identify best files to keep.")

        except Exception as e:
            print(f"Error processing {scoreboard_file}: {e}")

    # --- 2. Delete non-essential files from run/results ---
    results_dir = os.path.join(project_root, "run", "results")
    if not os.path.isdir(results_dir):
        print("Results directory not found. Nothing to clean.")
        return

    if not all_best_files:
        print("\nWarning: No best result files were identified. No files will be deleted.")
        return True

    print(f"\nIdentified {len(all_best_files)} best result file(s) to keep.")
    
    deleted_count = 0
    kept_count = 0
    for dirpath, _, filenames in os.walk(results_dir):
        for filename in filenames:
            file_path_abs = os.path.normpath(os.path.join(dirpath, filename))
            if file_path_abs not in all_best_files:
                try:
                    os.remove(file_path_abs)
                    # print(f"  - Deleted: {os.path.relpath(file_path_abs, project_root)}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  - Error deleting file {file_path_abs}: {e}")
            else:
                kept_count += 1

    print(f"\nCleanup complete. Kept {kept_count} file(s), deleted {deleted_count} file(s).")
    return True