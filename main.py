import os
import sys
import subprocess

def main():
    """
    Displays a menu for the user to select which model script to run or to clean history.
    """
    scripts = {
        '1': 'XGBoost_MultiLabel.py',
        '2': 'CNN_MultiLabel.py',
        '3': 'Transformer_MultiLabel.py',
        '4': 'CNN_ResNet.py'
    }

    while True:
        print("\n=============================================")
        print("  蛋白質轉譯後修飾位點預測模型")
        print("=============================================")
        print("請選擇要執行的操作：")
        print("  1: 運行 XGBoost 模型")
        print("  2: 運行 CNN 模型 (較佳)")
        print("  3: 運行 Transformer 模型")
        print("  4: 運行 CNN-ResNet 模型 (實驗性)")
        print("---------------------------------------------")
        print("  c: 清理歷史紀錄 (Clean execution history)")
        print("  q: 退出 (Quit)")
        print("=============================================")
        
        choice = input("請輸入您的選擇 (1/2/3/4/c/q): ").strip().lower()

        if choice in scripts:
            script_name = scripts[choice]
            script_path = os.path.abspath(os.path.join('scripts', script_name))
            interpreter = sys.executable
            
            command = [interpreter, script_path]

            # Special handling for XGBoost to select feature mode
            if choice == '1':
                print("---------------------------------------------")
                feature_choice = input("請選擇 XGBoost 特徵模式 (1) 全量特徵 (2) 精英特徵選擇: ").strip()
                if feature_choice == '2':
                    command.append("--mode")
                    command.append("select")
                    print("已選擇「精英特徵選擇」模式。")
                else:
                    command.append("--mode")
                    command.append("all")
                    print("已選擇「全量特徵」模式。")
                print("---------------------------------------------")

            print(f"\n--- 正在執行 {script_name} ---\n")
            
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # Print stdout and stderr from the script
            if result.stdout:
                print("--- 腳本標準輸出 (stdout) ---")
                print(result.stdout)
            if result.stderr:
                print("--- 腳本錯誤輸出 (stderr) ---")
                print(result.stderr)
            
            print(f"\n--- 腳本 {script_name} 執行完畢，退出碼: {result.returncode} ---\n")
            
            cont = input("按 Enter 鍵返回主選單，或輸入 'q' 退出: ").strip().lower()
            if cont == 'q':
                print("程式退出。")
                break

        elif choice == 'c':
            confirm = input("警告：此操作將刪除 'run/results/' 中除了每個任務最佳分數外的所有檔案。確定要繼續嗎？(y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    # We need to add project root to path to find util
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = script_dir # In this case, main.py is in the root
                    if project_root not in sys.path:
                        sys.path.append(project_root)
                    from util.scoreboard_manager import clean_history
                    print("\n--- 正在清理歷史紀錄 ---")
                    clean_history()
                except ImportError:
                    print("錯誤：找不到 'util.scoreboard_manager' 模組。")
                except Exception as e:
                    print(f"清理過程中發生錯誤: {e}")
            else:
                print("操作已取消。")
            input("\n按 Enter 鍵返回主選單...")

        elif choice == 'q':
            print("程式退出。")
            break
            
        else:
            print("無效的選擇，請重新輸入。")

if __name__ == "__main__":
    main()