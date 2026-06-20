# social_task

当前阶段暂不实现社科下游任务，只保留目录作为后续接口。

后续建议添加：

```text
covid_shift_analysis.py
seasonal_sales_analysis.py
style_preference_analysis.py
amazon_user_profile_analysis.py
visualization.py
common_stats.py
```

这些脚本应优先读取：

```text
social_output/model_outputs/{model_type}/item_style_probs.parquet
social_output/model_outputs/{model_type}/style_sales_time_series.parquet
social_output/model_outputs/{model_type}/user_time_preferences.parquet
social_output/model_outputs/{model_type}/knowledge_diagnostics.parquet
```
