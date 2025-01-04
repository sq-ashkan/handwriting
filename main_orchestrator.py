import shutil
from pathlib import Path

class MainOrchestrator:
    @staticmethod
    def execute_pipeline():
        try:
            from main_processor import main as process_main
            from analyse import main as analyze_main
            from main_data_enhancers import main as enhance_main
            
            process_main()
            analyze_main()
            
            if Path("analyse_result.json").exists():
                shutil.move("analyse_result.json", "analyse_result_before.json")
                
            enhance_main()
            analyze_main()
            
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True

if __name__ == "__main__":
    success = MainOrchestrator.execute_pipeline()
    exit(0 if success else 1)