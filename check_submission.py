import argparse
import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validate_submission(submission_path: Path, sample_path: Path) -> bool:
    """
    Валидация файла сабмита согласно требованиям соревнования.
    
    Проверяет:
    1. Наличие файлов.
    2. Схему колонок.
    3. Отсутствие NaN и бесконечностей в score.
    4. Отсутствие дубликатов (feed_id, item_id).
    5. Полное соответствие шаблону (sample_submission) по количеству и самим парам.
    """
    if not submission_path.exists():
        logger.error(f"Файл сабмита не найден: {submission_path}")
        return False
    if not sample_path.exists():
        logger.error(f"Файл шаблона не найден: {sample_path}")
        return False

    try:
        logger.info(f"Загрузка данных из {submission_path.name}...")
        sub = pl.read_parquet(submission_path)
        sample = pl.read_parquet(sample_path)
    except Exception as e:
        logger.error(f"Ошибка чтения Parquet-файлов: {e}")
        return False

    expected_columns = ["feed_id", "item_id", "score"]
    if sub.columns != expected_columns:
        logger.error(f"Неверные колонки. Ожидалось: {expected_columns}, получено: {sub.columns}")
        return False

    expected_schema = {
        "feed_id": pl.Int32,
        "item_id": pl.Int32,
        "score": pl.Float32
    }
    for col, dtype in expected_schema.items():
        if sub[col].dtype != dtype:
            logger.warning(f"Тип данных колонки {col} отличается от ожидаемого ({sub[col].dtype} != {dtype}).")

    invalid_scores = sub.filter(
        pl.col("score").is_null() | 
        pl.col("score").is_nan() | 
        pl.col("score").is_infinite()
    )
    if invalid_scores.height > 0:
        logger.error(f"Найдено {invalid_scores.height} некорректных значений в score (NaN, Null или бесконечность).")
        return False

    if sub.height != sub.unique(subset=["feed_id", "item_id"]).height:
        logger.error("Найдены дубликаты пар (feed_id, item_id).")
        return False

    if sub.height != sample.height:
        logger.error(f"Несовпадение количества строк. Сабмит: {sub.height}, Шаблон: {sample.height}")
        return False

    missing_pairs = sample.select(["feed_id", "item_id"]).join(
        sub.select(["feed_id", "item_id"]), 
        on=["feed_id", "item_id"], 
        how="anti"
    )
    if missing_pairs.height > 0:
        logger.error(f"В сабмите отсутствуют {missing_pairs.height} пар из шаблона.")
        return False

    logger.info("Валидация успешно пройдена. Файл сабмита полностью корректен.")
    return True

def main() -> None:
    parser = argparse.ArgumentParser(description="Валидация файла сабмита для соревнования Avito.")
    parser.add_argument(
        "--sub", 
        type=Path, 
        default=Path("runs/catboost-01/submission.parquet"), 
        help="Путь к файлу сабмита (по умолчанию: runs/catboost-01/submission.parquet)"
    )
    parser.add_argument(
        "--sample", 
        type=Path, 
        default=Path("data/sample_submission.parquet"), 
        help="Путь к файлу шаблона (по умолчанию: data/sample_submission.parquet)"
    )
    args = parser.parse_args()

    success = validate_submission(args.sub, args.sample)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()