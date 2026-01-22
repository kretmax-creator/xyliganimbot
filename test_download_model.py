"""
Тест загрузки embedding-модели.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.logging import setup_logging, get_logger
from src.model_loader import download_model

setup_logging(level="INFO", log_file="logs/app.log", audit_file="logs/audit.log", 
              log_user_messages=False, max_bytes=10485760, backup_count=5)

logger = get_logger(__name__)

def main():
    print("=" * 60)
    print("ТЕСТ: Загрузка embedding-модели")
    print("=" * 60)
    
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    models_dir = project_root / "models"
    
    print(f"Модель: {model_name}")
    print(f"Директория: {models_dir}")
    print("\n[INFO] Загрузка может занять несколько минут...")
    
    success = download_model(model_name=model_name, models_dir=models_dir)
    
    if success:
        print("[OK] Модель успешно загружена!")
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"   - Путь: {model_path}")
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            print(f"   - Размер: {total_size / (1024*1024):.2f} MB")
        return 0
    else:
        print("[ERROR] Ошибка при загрузке модели")
        return 1

if __name__ == "__main__":
    sys.exit(main())
