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
        '3': 'Transformer_MultiLabel.py'
    }

    while True:
        print("\n=============================================")
        print("  蛋白質轉譯後修飾位點預測模型")
        print("=============================================")
        print("請選擇要執行的操作：")
        print("  1: 運行 XGBoost 模型")
        print("  2: 運行 CNN 模型")
        print("  3: 運行 Transformer 模型")
        print("---------------------------------------------")
        print("  c: 清理歷史紀錄 (Clean execution history)")
        print("  q: 退出 (Quit)")
        print("=============================================")
        
        choice = input("請輸入您的選擇 (1/2/3/c/q): ").strip().lower()

        if choice in scripts:
            script_name = scripts[choice]
            # Use an absolute path for the script to be executed
            script_path = os.path.abspath(os.path.join('scripts', script_name))
            
            interpreter = sys.executable

            print(f"\n--- 正在執行 {script_name} ---\n")
            
            # Use subprocess.run for better error handling and output capture
            result = subprocess.run([interpreter, script_path], capture_output=True, text=True, encoding='utf-8')
            
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
            confirm = input("警告：此操作將刪除 'run/results/' 中除了每個任務最佳分數外的所有檔案，並清空計分板的執行歷史。確定要繼續嗎？(y/n): ").strip().lower()
            if confirm == 'y':
                try:
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

