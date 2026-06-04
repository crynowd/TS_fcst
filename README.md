## ICDM paper reproducibility

For the ICDM paper reproducibility package, start here:

[`paper_icdm/README.md`](paper_icdm/README.md)

This layer identifies the configs, processed data artifacts, benchmark outputs, meta-learning results, generated tables, and generated figures used for the paper.

Quick validation:

```bash
python paper_icdm/scripts/check_artifacts.py
python paper_icdm/scripts/build_paper_tables.py
python paper_icdm/scripts/build_paper_figures.py
```

The older smoke/demo configs are retained for lightweight development checks and are not the main paper benchmark.

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

### Paper route

The ICDM paper route is documented in [`paper_icdm/README.md`](paper_icdm/README.md). Use that entry point for artifact validation, generated paper tables, and generated paper figures.

### Development smoke/demo path

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

**4. Эксперимент прогнозирования**

```bash
python -m src.cli.run_forecasting_benchmark --config configs/forecasting_benchmark_smoke_v1.yaml
```

This command uses `configs/forecasting_benchmark_smoke_v1.yaml`, which is a development smoke/demo configuration. It is not the main paper benchmark; the paper benchmark is `configs/forecasting_benchmark_v2.yaml` and is documented through `paper_icdm/README.md`.

→ `artifacts/forecasting/series_metrics_smoke_v1.parquet`

---

**5. Подбор архитектур**

```bash
python -m src.cli.run_architecture_tuning_benchmark --config configs/architecture_tuning_benchmark_v1.yaml
```

→ `artifacts/architecture_tuning/*.parquet`, `reports/*.xlsx`

---

**6. Метамоделирование**

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
* на длинных горизонтах простые модели могут выигрывать по RMSE;
* хаотические модели не дают устойчивого преимущества, но полезны в отдельных случаях;

---

## Дополнительно

Подробное описание этапов, визуализации и анализ результатов представлены в ноутбуке:

[Исследовательский ноутбук](./TS_fcst_research.ipynb)
