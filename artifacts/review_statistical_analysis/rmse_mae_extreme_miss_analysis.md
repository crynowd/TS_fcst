# Additional RMSE/MAE and Extreme-Error Robustness Analysis

## 1. Цель анализа

Проверить два вопроса рецензента: (i) насколько выбор RMSE вместо MAE влияет на интерпретацию численной ошибки; (ii) не является ли сильная позиция `naive_zero` по RMSE артефактом нескольких крупных выбросов. Модели не переобучались; использованы только сохранённые forecast-level и metric-level артефакты.

## 2. Использованные файлы

- `artifacts\forecasting\forecasting_benchmark_v2\metrics_long.parquet`

- `artifacts\forecasting\forecasting_benchmark_v2\predictions.parquet`


Указанные в постановке `paired_best_config_routing_rows.csv`, `routing_rows_v2.parquet` и `best_config_per_task_v2.csv` не требовались для данного расчёта, потому что задачи A/B определяются уже сохранёнными fixed-model метриками и forecast-level прогнозами.

## 3. Краткая методика

- RMSE и MAE усреднены по `horizon × model_name` на строках `status=success` из `metrics_long.parquet`. Это fold-level среднее, соответствующее исходной таблице метрик.

- Ранги считаются отдельно внутри каждого horizon; меньшие RMSE/MAE получают лучший ранг. Корреляции RMSE/MAE рассчитаны по набору моделей внутри каждого horizon.

- Forecast-level ошибки рассчитаны как `error = y_true - y_pred`, `abs_error = abs(error)`, `squared_error = error^2`. Pooled forecast-level RMSE может отличаться от среднего fold-level RMSE, потому что меняет веса наблюдений.

- Для top 1%/5% squared errors рассчитана доля SSE; trimming удаляет соответствующую верхнюю долю squared errors, winsorization ограничивает squared error пороговым значением верхней доли.

- Miss probability рассчитана как доля прогнозов с `abs_error > k × MAD(y_true)` для `k in {1,2,3}`. MAPE не использовалась, поскольку для доходностей около нуля она некорректна. Train-based `MAD(y_true)` невозможно восстановить из `predictions.parquet`; поэтому пороги рассчитаны по распределению `y_true` внутри соответствующего horizon.

## 4. RMSE/MAE mean by model × horizon

| horizon | model_name | rmse_mean | mae_mean | rmse_rank | mae_rank | n_metric_rows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | naive_zero | 0.0284285 | 0.0166137 | 1 | 1 | 1254 |
| 1 | naive_mean | 0.0284403 | 0.0166566 | 2 | 2 | 1254 |
| 1 | lstm_forecast | 0.0284878 | 0.0167349 | 3 | 4 | 1254 |
| 1 | chaotic_lstm_forecast | 0.0284937 | 0.016719 | 4 | 3 | 1254 |
| 1 | ridge_lag | 0.0287662 | 0.0170875 | 5 | 5 | 1254 |
| 1 | vanilla_mlp | 0.0288669 | 0.0173581 | 6 | 8 | 1254 |
| 1 | transient_chaotic_esn | 0.0289102 | 0.0171038 | 7 | 7 | 1254 |
| 1 | esn | 0.0289127 | 0.0171018 | 8 | 6 | 1254 |
| 1 | chaotic_mlp | 0.0294915 | 0.0180705 | 9 | 9 | 1254 |
| 1 | chaotic_logistic_net | 0.03137 | 0.020509 | 10 | 10 | 1254 |
| 1 | chaotic_esn | 0.0345058 | 0.0212339 | 11 | 11 | 1254 |
| 5 | naive_zero | 0.0595196 | 0.0375815 | 1 | 1 | 1254 |
| 5 | naive_mean | 0.0596509 | 0.0377881 | 2 | 2 | 1254 |
| 5 | lstm_forecast | 0.059885 | 0.0380288 | 3 | 3 | 1254 |
| 5 | chaotic_lstm_forecast | 0.0599812 | 0.038045 | 4 | 4 | 1254 |
| 5 | vanilla_mlp | 0.0607763 | 0.0390869 | 5 | 5 | 1254 |
| 5 | chaotic_mlp | 0.0607844 | 0.0392441 | 6 | 7 | 1254 |
| 5 | ridge_lag | 0.0608723 | 0.0391505 | 7 | 6 | 1254 |
| 5 | esn | 0.0611841 | 0.039384 | 8 | 8 | 1254 |
| 5 | transient_chaotic_esn | 0.0614667 | 0.0397029 | 9 | 9 | 1254 |
| 5 | chaotic_logistic_net | 0.0623424 | 0.0411705 | 10 | 10 | 1254 |
| 5 | chaotic_esn | 0.0706217 | 0.0462181 | 11 | 11 | 1254 |
| 20 | naive_zero | 0.112945 | 0.0793293 | 1 | 1 | 1254 |
| 20 | naive_mean | 0.113931 | 0.0805598 | 2 | 2 | 1254 |
| 20 | lstm_forecast | 0.114285 | 0.0807724 | 3 | 3 | 1254 |
| 20 | chaotic_lstm_forecast | 0.115011 | 0.0813845 | 4 | 4 | 1254 |
| 20 | ridge_lag | 0.116186 | 0.0828141 | 5 | 6 | 1254 |
| 20 | vanilla_mlp | 0.116559 | 0.0827908 | 6 | 5 | 1254 |
| 20 | esn | 0.117051 | 0.0833763 | 7 | 7 | 1254 |
| 20 | chaotic_mlp | 0.118274 | 0.0851545 | 8 | 9 | 1254 |
| 20 | transient_chaotic_esn | 0.119251 | 0.0847757 | 9 | 8 | 1254 |
| 20 | chaotic_logistic_net | 0.119614 | 0.0865189 | 10 | 10 | 1254 |
| 20 | chaotic_esn | 0.125896 | 0.0877041 | 11 | 11 | 1254 |

## 5. RMSE vs MAE ranking summary

| horizon | n_models | best_by_rmse | best_by_mae | best_same | spearman_rank_rmse_vs_mae | spearman_rank_pvalue | pearson_value_rmse_vs_mae | pearson_value_pvalue | spearman_value_rmse_vs_mae | spearman_value_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11 | naive_zero | naive_zero | true | 0.954545 | 4.9889e-06 | 0.955375 | 4.59769e-06 | 0.954545 | 4.9889e-06 |
| 5 | 11 | naive_zero | naive_zero | true | 0.990909 | 3.76257e-09 | 0.985665 | 2.89956e-08 | 0.990909 | 3.76257e-09 |
| 20 | 11 | naive_zero | naive_zero | true | 0.981818 | 8.40307e-08 | 0.940859 | 1.59858e-05 | 0.981818 | 8.40307e-08 |

### Best fixed model check for h = 1, 5, 20

| horizon | best_by_rmse | best_by_mae | best_same | spearman_rank_rmse_vs_mae | pearson_value_rmse_vs_mae | spearman_value_rmse_vs_mae |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | naive_zero | naive_zero | true | 0.954545 | 0.955375 | 0.954545 |
| 5 | naive_zero | naive_zero | true | 0.990909 | 0.985665 | 0.990909 |
| 20 | naive_zero | naive_zero | true | 0.981818 | 0.940859 | 0.981818 |

## 6. Extreme error contribution summary

| horizon | model_name | pooled_forecast_rmse | pooled_forecast_mae | top_1pct_sse_share | top_5pct_sse_share | rmse_trim_top_1pct | rmse_trim_top_5pct | rmse_winsor_top_1pct | rmse_winsor_top_5pct | n_forecasts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lstm_forecast | 0.0462942 | 0.0167156 | 0.773457 | 0.876054 | 0.0221454 | 0.0167218 | 0.0247772 | 0.0201455 | 599844 |
| 1 | transient_chaotic_esn | 0.0463404 | 0.0170758 | 0.764717 | 0.871627 | 0.0225911 | 0.0170347 | 0.0253042 | 0.0205372 | 599844 |
| 1 | esn | 0.0464452 | 0.0170752 | 0.765475 | 0.87206 | 0.0226057 | 0.0170445 | 0.0253273 | 0.0205527 | 599844 |
| 1 | naive_zero | 0.0466674 | 0.0165945 | 0.778698 | 0.879241 | 0.0220642 | 0.0166384 | 0.0246861 | 0.020078 | 599844 |
| 1 | naive_mean | 0.0466752 | 0.016637 | 0.778449 | 0.878998 | 0.0220803 | 0.0166579 | 0.0246993 | 0.0200868 | 599844 |
| 1 | chaotic_lstm_forecast | 0.0466842 | 0.0166994 | 0.777857 | 0.878506 | 0.0221141 | 0.016695 | 0.0247432 | 0.0201173 | 599844 |
| 1 | ridge_lag | 0.0469769 | 0.0170581 | 0.771141 | 0.874456 | 0.0225866 | 0.0170773 | 0.0252614 | 0.0205614 | 599844 |
| 1 | chaotic_mlp | 0.0473423 | 0.0180493 | 0.761845 | 0.863014 | 0.02322 | 0.0179773 | 0.025763 | 0.0213154 | 599844 |
| 1 | vanilla_mlp | 0.0477841 | 0.0173311 | 0.775021 | 0.876179 | 0.0227791 | 0.0172511 | 0.0254857 | 0.0207414 | 599844 |
| 1 | chaotic_logistic_net | 0.0484602 | 0.0205166 | 0.723898 | 0.825898 | 0.0255919 | 0.0207456 | 0.0279289 | 0.0239683 | 599844 |
| 1 | chaotic_esn | 0.0519568 | 0.0211688 | 0.709298 | 0.843196 | 0.0281546 | 0.0211086 | 0.0316454 | 0.025477 | 599844 |
| 5 | naive_zero | 0.0796179 | 0.0375345 | 0.603315 | 0.78056 | 0.0503983 | 0.0382655 | 0.0560475 | 0.0460831 | 607368 |
| 5 | naive_mean | 0.0797272 | 0.037738 | 0.601837 | 0.779251 | 0.0505615 | 0.0384321 | 0.0562211 | 0.0462551 | 607368 |
| 5 | lstm_forecast | 0.0800365 | 0.03798 | 0.601528 | 0.778521 | 0.0507773 | 0.0386449 | 0.0564277 | 0.0464521 | 607368 |
| 5 | chaotic_lstm_forecast | 0.0801542 | 0.037995 | 0.602285 | 0.779334 | 0.0508037 | 0.0386307 | 0.0564995 | 0.0464606 | 607368 |
| 5 | chaotic_mlp | 0.0805842 | 0.0391876 | 0.586544 | 0.766931 | 0.0520772 | 0.0399145 | 0.0577057 | 0.047803 | 607368 |
| 5 | ridge_lag | 0.0806935 | 0.0390809 | 0.586173 | 0.768309 | 0.0521712 | 0.0398503 | 0.057886 | 0.0478446 | 607368 |
| 5 | esn | 0.0808827 | 0.0393126 | 0.584028 | 0.766795 | 0.0524288 | 0.040074 | 0.0581692 | 0.0481059 | 607368 |
| 5 | vanilla_mlp | 0.0809689 | 0.0390268 | 0.592038 | 0.771406 | 0.0519769 | 0.0397182 | 0.0576839 | 0.0476284 | 607368 |
| 5 | transient_chaotic_esn | 0.0813764 | 0.0396326 | 0.582822 | 0.765569 | 0.0528253 | 0.0404245 | 0.0585745 | 0.0485085 | 607368 |
| 5 | chaotic_logistic_net | 0.0817277 | 0.0411418 | 0.576075 | 0.75269 | 0.0534807 | 0.0416993 | 0.0589669 | 0.0494421 | 607368 |
| 5 | chaotic_esn | 0.0929312 | 0.0461155 | 0.553665 | 0.757069 | 0.0623985 | 0.0469939 | 0.0696076 | 0.0569077 | 607368 |
| 20 | naive_zero | 0.142821 | 0.0792254 | 0.462493 | 0.682501 | 0.105237 | 0.0825661 | 0.115067 | 0.0981007 | 602418 |
| 20 | naive_mean | 0.143743 | 0.0804341 | 0.457229 | 0.677141 | 0.106434 | 0.0837977 | 0.116213 | 0.0994016 | 602418 |
| 20 | lstm_forecast | 0.143906 | 0.0806622 | 0.455091 | 0.676053 | 0.106764 | 0.0840338 | 0.116634 | 0.09973 | 602418 |
| 20 | ridge_lag | 0.144419 | 0.0826628 | 0.43281 | 0.661643 | 0.109313 | 0.0861889 | 0.119438 | 0.102111 | 602418 |
| 20 | chaotic_lstm_forecast | 0.144894 | 0.08127 | 0.45478 | 0.676361 | 0.107528 | 0.0845707 | 0.117537 | 0.100351 | 602418 |
| 20 | vanilla_mlp | 0.146069 | 0.0826523 | 0.447854 | 0.670433 | 0.109086 | 0.0860336 | 0.119139 | 0.101949 | 602418 |
| 20 | chaotic_mlp | 0.147412 | 0.085039 | 0.436512 | 0.656377 | 0.111214 | 0.088657 | 0.12103 | 0.104375 | 602418 |
| 20 | esn | 0.147649 | 0.0832094 | 0.452034 | 0.672633 | 0.109848 | 0.0866736 | 0.119994 | 0.102651 | 602418 |
| 20 | chaotic_logistic_net | 0.148695 | 0.0864165 | 0.433064 | 0.650851 | 0.112524 | 0.0901444 | 0.122252 | 0.105877 | 602418 |
| 20 | transient_chaotic_esn | 0.15141 | 0.0845957 | 0.45991 | 0.680416 | 0.111833 | 0.0878186 | 0.122451 | 0.104221 | 602418 |
| 20 | chaotic_esn | 0.170527 | 0.0874463 | 0.54725 | 0.737554 | 0.11532 | 0.0896296 | 0.126833 | 0.106741 | 602418 |

## 7. Trimming/winsorization winners

| horizon | criterion | winner_model | winner_value | naive_zero_value | naive_zero_rank | naive_zero_is_winner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Pooled forecast-level RMSE | lstm_forecast | 0.0462942 | 0.0466674 | 4 | false |
| 1 | Pooled forecast-level MAE | naive_zero | 0.0165945 | 0.0165945 | 1 | true |
| 1 | RMSE trimmed top 1% SE | naive_zero | 0.0220642 | 0.0220642 | 1 | true |
| 1 | RMSE trimmed top 5% SE | naive_zero | 0.0166384 | 0.0166384 | 1 | true |
| 1 | RMSE winsorized top 1% SE | naive_zero | 0.0246861 | 0.0246861 | 1 | true |
| 1 | RMSE winsorized top 5% SE | naive_zero | 0.020078 | 0.020078 | 1 | true |
| 5 | Pooled forecast-level RMSE | naive_zero | 0.0796179 | 0.0796179 | 1 | true |
| 5 | Pooled forecast-level MAE | naive_zero | 0.0375345 | 0.0375345 | 1 | true |
| 5 | RMSE trimmed top 1% SE | naive_zero | 0.0503983 | 0.0503983 | 1 | true |
| 5 | RMSE trimmed top 5% SE | naive_zero | 0.0382655 | 0.0382655 | 1 | true |
| 5 | RMSE winsorized top 1% SE | naive_zero | 0.0560475 | 0.0560475 | 1 | true |
| 5 | RMSE winsorized top 5% SE | naive_zero | 0.0460831 | 0.0460831 | 1 | true |
| 20 | Pooled forecast-level RMSE | naive_zero | 0.142821 | 0.142821 | 1 | true |
| 20 | Pooled forecast-level MAE | naive_zero | 0.0792254 | 0.0792254 | 1 | true |
| 20 | RMSE trimmed top 1% SE | naive_zero | 0.105237 | 0.105237 | 1 | true |
| 20 | RMSE trimmed top 5% SE | naive_zero | 0.0825661 | 0.0825661 | 1 | true |
| 20 | RMSE winsorized top 1% SE | naive_zero | 0.115067 | 0.115067 | 1 | true |
| 20 | RMSE winsorized top 5% SE | naive_zero | 0.0981007 | 0.0981007 | 1 | true |

## 8. Miss probability winners and ranking summary

### Winners

| horizon | criterion | winner_model | winner_value | naive_zero_value | naive_zero_rank | naive_zero_is_winner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | miss_probability_1mad | naive_zero | 0.499998 | 0.499998 | 1 | true |
| 1 | miss_probability_2mad | naive_zero | 0.264485 | 0.264485 | 1 | true |
| 1 | miss_probability_3mad | naive_zero | 0.15256 | 0.15256 | 1 | true |
| 5 | miss_probability_1mad | naive_zero | 0.500258 | 0.500258 | 1 | true |
| 5 | miss_probability_2mad | naive_zero | 0.262666 | 0.262666 | 1 | true |
| 5 | miss_probability_3mad | naive_zero | 0.150233 | 0.150233 | 1 | true |
| 20 | miss_probability_1mad | naive_zero | 0.501205 | 0.501205 | 1 | true |
| 20 | miss_probability_2mad | naive_zero | 0.258014 | 0.258014 | 1 | true |
| 20 | miss_probability_3mad | naive_zero | 0.142932 | 0.142932 | 1 | true |

### Ranking summary

| horizon | model_name | rmse_rank | mae_rank | miss_probability_1mad | miss_rank_1mad | miss_probability_2mad | miss_rank_2mad | miss_probability_3mad | miss_rank_3mad | n_forecasts | threshold_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | naive_zero | 1 | 1 | 0.499998 | 1 | 0.264485 | 1 | 0.15256 | 1 | 599844 | test_y_true_by_horizon |
| 1 | naive_mean | 2 | 2 | 0.500645 | 2 | 0.264879 | 2 | 0.152778 | 2 | 599844 | test_y_true_by_horizon |
| 1 | chaotic_lstm_forecast | 4 | 3 | 0.502929 | 3 | 0.265714 | 3 | 0.153313 | 3 | 599844 | test_y_true_by_horizon |
| 1 | lstm_forecast | 3 | 4 | 0.503409 | 4 | 0.266309 | 4 | 0.1536 | 4 | 599844 | test_y_true_by_horizon |
| 1 | esn | 8 | 6 | 0.510464 | 6 | 0.271717 | 5 | 0.157876 | 6 | 599844 | test_y_true_by_horizon |
| 1 | transient_chaotic_esn | 7 | 7 | 0.510431 | 5 | 0.271734 | 6 | 0.157578 | 5 | 599844 | test_y_true_by_horizon |
| 1 | ridge_lag | 5 | 5 | 0.510676 | 7 | 0.273091 | 7 | 0.158725 | 7 | 599844 | test_y_true_by_horizon |
| 1 | vanilla_mlp | 6 | 8 | 0.518563 | 8 | 0.277856 | 8 | 0.160767 | 8 | 599844 | test_y_true_by_horizon |
| 1 | chaotic_mlp | 9 | 9 | 0.563446 | 9 | 0.302705 | 9 | 0.170469 | 9 | 599844 | test_y_true_by_horizon |
| 1 | chaotic_esn | 11 | 11 | 0.581888 | 10 | 0.342847 | 10 | 0.213474 | 10 | 599844 | test_y_true_by_horizon |
| 1 | chaotic_logistic_net | 10 | 10 | 0.626239 | 11 | 0.398595 | 11 | 0.222505 | 11 | 599844 | test_y_true_by_horizon |
| 5 | naive_zero | 1 | 1 | 0.500258 | 1 | 0.262666 | 1 | 0.150233 | 1 | 607368 | test_y_true_by_horizon |
| 5 | naive_mean | 2 | 2 | 0.503698 | 2 | 0.263983 | 2 | 0.151305 | 2 | 607368 | test_y_true_by_horizon |
| 5 | lstm_forecast | 3 | 3 | 0.507325 | 4 | 0.266381 | 3 | 0.152237 | 4 | 607368 | test_y_true_by_horizon |
| 5 | chaotic_lstm_forecast | 4 | 4 | 0.506576 | 3 | 0.266397 | 4 | 0.152203 | 3 | 607368 | test_y_true_by_horizon |
| 5 | vanilla_mlp | 5 | 5 | 0.521478 | 6 | 0.275713 | 5 | 0.158746 | 5 | 607368 | test_y_true_by_horizon |
| 5 | ridge_lag | 7 | 6 | 0.518689 | 5 | 0.276602 | 6 | 0.160158 | 7 | 607368 | test_y_true_by_horizon |
| 5 | chaotic_mlp | 6 | 7 | 0.525067 | 9 | 0.278498 | 7 | 0.160002 | 6 | 607368 | test_y_true_by_horizon |
| 5 | esn | 8 | 8 | 0.521565 | 7 | 0.278935 | 8 | 0.16155 | 8 | 607368 | test_y_true_by_horizon |
| 5 | transient_chaotic_esn | 9 | 9 | 0.524807 | 8 | 0.281498 | 9 | 0.163726 | 9 | 607368 | test_y_true_by_horizon |
| 5 | chaotic_logistic_net | 10 | 10 | 0.56837 | 11 | 0.300113 | 10 | 0.169474 | 10 | 607368 | test_y_true_by_horizon |
| 5 | chaotic_esn | 11 | 11 | 0.564658 | 10 | 0.325279 | 11 | 0.200715 | 11 | 607368 | test_y_true_by_horizon |
| 20 | naive_zero | 1 | 1 | 0.501205 | 1 | 0.258014 | 1 | 0.142932 | 1 | 602418 | test_y_true_by_horizon |
| 20 | naive_mean | 2 | 2 | 0.510584 | 3 | 0.263618 | 2 | 0.146003 | 2 | 602418 | test_y_true_by_horizon |
| 20 | lstm_forecast | 3 | 3 | 0.509965 | 2 | 0.264715 | 3 | 0.146815 | 3 | 602418 | test_y_true_by_horizon |
| 20 | chaotic_lstm_forecast | 4 | 4 | 0.513262 | 4 | 0.266699 | 4 | 0.148648 | 4 | 602418 | test_y_true_by_horizon |
| 20 | vanilla_mlp | 6 | 5 | 0.521425 | 6 | 0.27307 | 5 | 0.152688 | 5 | 602418 | test_y_true_by_horizon |
| 20 | ridge_lag | 5 | 6 | 0.521211 | 5 | 0.273292 | 6 | 0.152949 | 6 | 602418 | test_y_true_by_horizon |
| 20 | esn | 7 | 7 | 0.523522 | 7 | 0.275345 | 7 | 0.154316 | 7 | 602418 | test_y_true_by_horizon |
| 20 | transient_chaotic_esn | 9 | 8 | 0.525975 | 8 | 0.27878 | 8 | 0.157489 | 8 | 602418 | test_y_true_by_horizon |
| 20 | chaotic_esn | 11 | 11 | 0.529813 | 9 | 0.283723 | 9 | 0.162475 | 10 | 602418 | test_y_true_by_horizon |
| 20 | chaotic_mlp | 8 | 9 | 0.5409 | 10 | 0.288524 | 10 | 0.160412 | 9 | 602418 | test_y_true_by_horizon |
| 20 | chaotic_logistic_net | 10 | 10 | 0.551107 | 11 | 0.295916 | 11 | 0.165038 | 11 | 602418 | test_y_true_by_horizon |

## 9. Готовый текстовый вывод для ответа рецензенту

Для численной ошибки RMSE использовалась как стандартная magnitude-error метрика, которая сильнее штрафует крупные ошибки и поэтому соответствует задаче контроля больших промахов прогноза. В качестве робастной альтернативы была дополнительно рассчитана MAE. Во всех горизонтах лучшая fixed model по среднему fold-level RMSE совпадает с лучшей по среднему fold-level MAE; минимальная Spearman-корреляция рангов RMSE/MAE по горизонтам = 0.955. Это означает, что выбор RMSE вместо MAE не меняет основной вывод о best fixed model и общем ранжировании. Для проверки, не является ли преимущество naive_zero артефактом нескольких экстремальных наблюдений, были рассчитаны доли SSE верхних 1% и 5% squared errors, RMSE после trimming/winsorization этих наблюдений, а также miss_probability без MAPE с порогами на основе MAD. При pooled forecast-level пересчёте есть отличие от fold-level среднего: h=1, Pooled forecast-level RMSE: winner=lstm_forecast, naive_zero_rank=4. После trimming и winsorization верхних 1%/5% squared errors naive_zero остаётся winner во всех горизонтах. По miss_probability с порогами 1x/2x/3x MAD naive_zero также остаётся winner во всех горизонтах. Поэтому вывод о сильной позиции naive_zero не объясняется только несколькими экстремальными squared errors; при этом следует явно оговорить чувствительность RMSE к крупным ошибкам и ограничение post-hoc MAD-порогов.


## 10. Ограничения

- RMSE действительно чувствительна к экстремальным ошибкам; это свойство является причиной дополнительной проверки через MAE, trimming, winsorization и miss probability.

- MAE и miss probability используются как дополнительные робастные проверки, а не как результат переобучения или новая процедура выбора моделей.

- Train-based порог для miss probability не рассчитан, потому что `predictions.parquet` не содержит обучающего распределения `y_true` для каждого fold; доступен только `y_train_mean`. Порог `k × MAD` рассчитан по `y_true` внутри horizon, что следует трактовать как post-hoc sensitivity check с возможной test-distribution зависимостью.
