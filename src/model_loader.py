"""
Модуль загрузки embedding-моделей для xyliganimbot.

Обеспечивает загрузку моделей из HuggingFace и сохранение локально.
Процесс независим от загрузки контента и векторизации.
"""

from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from src.logging import get_logger

logger = get_logger(__name__)


def download_model(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    models_dir: Path = Path("models")
) -> bool:
    """
    Загружает embedding-модель из HuggingFace и сохраняет локально.

    Args:
        model_name: Имя модели для загрузки (по умолчанию paraphrase-multilingual-MiniLM-L12-v2)
        models_dir: Директория для сохранения моделей (по умолчанию models/)

    Returns:
        True если успешно, False при ошибке
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
        return False

    try:
        # Создаем директорию для моделей, если её нет
        models_dir.mkdir(parents=True, exist_ok=True)

        # Путь для сохранения модели
        model_path = models_dir / model_name

        # Проверяем, не загружена ли модель уже
        if model_path.exists():
            logger.info(f"Model already exists at {model_path}, skipping download")
            return True

        logger.info(f"Downloading embedding model from HuggingFace: {model_name}")
        logger.info(f"Target directory: {model_path}")

        # Загружаем модель из HuggingFace
        # SentenceTransformer автоматически скачивает модель при первом использовании
        model = SentenceTransformer(model_name)

        # Сохраняем модель локально
        model.save(str(model_path))
        logger.info(f"Model successfully saved to {model_path}")

        return True

    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}", exc_info=True)
        return False
