# Моделирование и прогнозирование временных рядов с использованием хаотических нейронных сетей

## Описание проекта

Проект посвящён исследованию методов прогнозирования финансовых временных рядов с акцентом на хаотические нейронные сети и их применимость.

В рамках работы:

* реализован набор моделей прогнозирования (классические и хаотические нейросетевые архитектуры);
* проведено их систематическое сравнение на различных горизонтах;
* исследована связь между характеристиками временных рядов и качеством моделей;
* построена метамодель, выбирающая наиболее подходящую модель для конкретного ряда.
---

## Запуск и воспроизведение экспериментов

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Этапы:

**1. Подготовка данных**

```bash
python -m src.cli.run_data_inventory --config configs/data_inventory_v1.yaml
python -m src.cli.run_log_returns_pipeline --config configs/data_inventory_v1.yaml
```

→ `artifacts/processed/log_returns_v1.parquet`, `dataset_profiles_v1.parquet`

---

**2. Расчёт признаков**

```bash
python -m src.cli.run_feature_block --block A --config configs/features_block_A_v1.yaml
python -m src.cli.run_feature_block --block B --config configs/features_block_B_v1.yaml
python -m src.cli.run_feature_block --block C --config configs/features_block_C_v1.yaml
python -m src.cli.run_feature_block --block D --config configs/features_block_D_v1.yaml
```

→ `artifacts/features/features_block_*.parquet`

---

**3. Отбор признаков**

```bash
python -m src.cli.run_feature_consolidation --config configs/feature_consolidation_v1.yaml
```

→ `clustering_features_base_v1.parquet`, `clustering_features_with_chaos_v1.parquet`

---

**4. Кластеризация**

```bash
python -m src.cli.run_clustering_experiments --config configs/clustering_experiments_v1.yaml
```

→ `artifacts/clustering/*.parquet`

---

**5. Эксперимент прогнозирования**

```bash
python -m src.cli.run_forecasting_benchmark --config configs/forecasting_benchmark_smoke_v1.yaml
```

→ `artifacts/forecasting/series_metrics_smoke_v1.parquet`

---

**6. Подбор архитектур**

```bash
python -m src.cli.run_architecture_tuning_benchmark --config configs/architecture_tuning_benchmark_v1.yaml
```

→ `artifacts/architecture_tuning/*.parquet`, `reports/*.xlsx`

---

**7. Метамоделирование**

```bash
python -m src.cli.run_meta_modeling --config configs/meta_modeling_v1.yaml
```

→ `artifacts/meta_modeling/*.parquet`, `reports/*.xlsx`

---

## Структура проекта

* `src/` — основной код проекта (модели, пайплайны, CLI)
* `src/forecasting/architectures/` — реализации моделей (ESN, LSTM, MLP, хаотические варианты)
* `configs/` — конфигурации всех экспериментов
* `artifacts/` — результаты (прогнозы, метрики, отчёты)
* `notebooks/` — исследовательский анализ и визуализации
* `tests/` — тесты

---

## Ключевые результаты

* универсальной лучшей модели не существует;
* ESN показывает лучшие результаты на коротких горизонтах;
* на длинных горизонтах простые модели могут выигрывать по RMSE;
* разные метрики приводят к различным выводам;
* хаотические модели не дают устойчивого преимущества, но полезны в отдельных случаях;

---

## Дополнительно

Подробное описание этапов, визуализации и анализ результатов представлены в ноутбуке:

[Исследовательский ноутбук](./TS_fcst_research.ipynb)
