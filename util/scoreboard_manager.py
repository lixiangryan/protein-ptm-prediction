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
        except (pd.errors.EmptyDataError, FileNotFoundError, IndexError):
            history_df = pd.DataFrame()

    # --- 2. Append new entry and sort ---
    history_df = pd.concat([history_df, new_score_entry], ignore_index=True)
    history_df = history_df.drop_duplicates(subset=['Timestamp', 'Method', 'Task'], keep='last')
    
    history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
    history_df = history_df.sort_values(by="Timestamp", ascending=False)
    history_df['Timestamp'] = history_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- 3. Prepare best scores DataFrame ---
    if 'RMSE_Score' in history_df.columns:
        score_col = 'RMSE_Score'
        sort_ascending = True
    elif 'Accuracy_Score' in history_df.columns:
        score_col = 'Accuracy_Score'
        sort_ascending = False
    else:
        try:
            with open(scoreboard_file, 'w', encoding='utf-8', newline='') as f:
                history_df.to_csv(f, header=True, index=False, lineterminator='\n')
            return True
        except IOError as e:
            print(f"寫入 {scoreboard_file} 時發生錯誤: {e}")
            return False

    history_df[score_col] = pd.to_numeric(history_df[score_col], errors='coerce')
    valid_scores_df = history_df.dropna(subset=[score_col])
    
    top_rows = []
    columns = new_score_entry.columns
    
    
    # Dynamically find all unique tasks from the history
    if not valid_scores_df.empty:
        top_tasks = valid_scores_df['Task'].unique()
    else:
        # If history is empty, use the task from the new entry
        top_tasks = new_score_entry['Task'].unique()

    for task in top_tasks:
        if not valid_scores_df.empty and task in valid_scores_df['Task'].values:
            task_df = valid_scores_df[valid_scores_df['Task'] == task]
            # Ensure score column is numeric for sorting
            task_df[score_col] = pd.to_numeric(task_df[score_col])
            task_best_score_row = task_df.sort_values(by=score_col, ascending=sort_ascending).iloc[0]
            top_rows.append(task_best_score_row.to_dict())
        else:
            placeholder_row = {col: '-' for col in columns}
            placeholder_row['Method'] = 'No score yet'
            placeholder_row['Task'] = task
            top_rows.append(placeholder_row)
            
    top_df = pd.DataFrame(top_rows, columns=columns)

    # --- 4. Rewrite the entire file ---
    try:
        with open(scoreboard_file, 'w', encoding='utf-8', newline='') as f:
            top_df.to_csv(f, header=True, index=False, lineterminator='\n')
            f.write('\n')
            history_df.to_csv(f, header=True, index=False, lineterminator='\n')
        return True
            
    except IOError as e:
        print(f"寫入 {scoreboard_file} 時發生錯誤: {e}")
        return False

def clean_history():
    """
    Cleans the history from scoreboards and deletes all but the best result files.
    """
    print("--- Starting History Cleanup ---")
    
    scoreboards_to_process = [
        os.path.join("run", "scoreboard_classify.csv"), # For XGBoost
        os.path.join("run", "scoreboard_cnn.csv"),      # For CNN
        os.path.join("run", "scoreboard_transformer.csv") # For Transformer
    ]
    
    all_best_files = set()

    # --- 1. Find best files and rewrite scoreboards ---
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
                print(f"Could not find history separator in {scoreboard_file}. Aborting cleanup for this file.")
                continue

            top_csv = StringIO("".join(lines[:blank_line_indices[0]]))
            top_df = pd.read_csv(top_csv)
            
            for file_path in top_df['OutputFile']:
                if isinstance(file_path, str) and file_path != '-':
                    # Normalize path separators for cross-platform compatibility
                    normalized_path = os.path.normpath(os.path.join(project_root, file_path))
                    all_best_files.add(normalized_path)

            # Rewrite the scoreboard with only the top summary
            with open(scoreboard_file, 'w', encoding='utf-8', newline='') as f:
                top_df.to_csv(f, header=True, index=False, lineterminator='\n')
                f.write('\n')
            print(f"Cleaned history in {scoreboard_file}.")

        except Exception as e:
            print(f"Error processing {scoreboard_file}: {e}")

    # --- 2. Delete non-essential files from run/results ---
    results_dir = os.path.join(project_root, "run", "results")
    if not os.path.isdir(results_dir):
        print("Results directory not found. Nothing to clean.")
        return

    print(f"\nIdentified {len(all_best_files)} best result file(s) to keep.")
    
    deleted_count = 0
    kept_count = 0
    for dirpath, _, filenames in os.walk(results_dir):
        for filename in filenames:
            file_path_abs = os.path.normpath(os.path.join(dirpath, filename))
            if file_path_abs not in all_best_files:
                try:
                    os.remove(file_path_abs)
                    print(f"  - Deleted: {os.path.relpath(file_path_abs, project_root)}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  - Error deleting file {file_path_abs}: {e}")
            else:
                kept_count += 1

    print(f"\nCleanup complete. Kept {kept_count} file(s), deleted {deleted_count} file(s).")
    return True

