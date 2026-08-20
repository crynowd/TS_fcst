# Feature stability summary

Feature stability is evaluated across rolling-origin folds using Pearson/Spearman correlation, absolute changes, coefficient of variation, and PSI.

Most stable features:
| feature | spearman_corr_mean | mean_abs_change | cv_mean | psi_mean | stability_score |
| --- | --- | --- | --- | --- | --- |
| tail_ratio_upper | 0.859099 | 1.62572 | 0.135781 | 0.168676 | 0.843876 |
| vr_q10 | 0.830729 | 0.10938 | 0.111288 | 0.114661 | 0.819432 |
| robust_kurtosis | 0.780198 | 0.106732 | 0.0512471 | 0.0772105 | 0.773776 |
| hurst_rs | 0.724028 | 0.0310634 | 0.0463 | 0.224135 | 0.710506 |
| hurst_dfa | 0.69606 | 0.0450886 | 0.0756136 | 0.209421 | 0.681808 |
| hill_tail_index | 0.693398 | 0.512605 | 0.136964 | 0.365435 | 0.668278 |
| kurtosis | 0.668485 | 13.758 | 0.499146 | 0.550952 | 0.61598 |
| abs_acf_lag_10 | 0.635309 | 0.0422099 | 1.00095 | 0.297461 | 0.570389 |

Least stable features:
| feature | spearman_corr_mean | mean_abs_change | cv_mean | psi_mean | stability_score |
| --- | --- | --- | --- | --- | --- |
| embedding_dimension | 0.0640931 | 1.93089 | 0.220522 | 0.29623 | 0.0382554 |
| selected_delay_tau | 0.198887 | 1.09304 | 0.251286 | 0.0541727 | 0.183614 |
| spectral_flatness | 0.559669 | 0.121734 | 0.115733 | 5.12589 | 0.297588 |
| acf_lag_2 | 0.608818 | 0.0379912 | 16.466 | 0.120311 | 0.310502 |
| abs_acf_lag_50 | 0.550411 | 0.0358582 | 4.34486 | 0.425111 | 0.311913 |
| correlation_dimension | 0.424201 | 0.851528 | 0.205259 | 0.144425 | 0.406716 |
| acf_lag_100 | 0.643045 | 0.0203353 | 3.35092 | 0.139386 | 0.46853 |
| abs_acf_lag_2 | 0.5095 | 0.0653649 | 0.320204 | 0.288151 | 0.479082 |