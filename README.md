# Stage 0: Data and Environment Preparation
```bash
conda create -n HM python=3.12
conda activate HM
pip install -r requirements.txt

bash scripts/prepare_hm.sh \
  --data_root ./data \
  --output_root ./output

bash scripts/prepare_amazon.sh \
  --data_root ./data \
  --time_filter_from_hm 1 \
  --review_sample_ratio 1.0
```

# Stage1: Amazon Review Retrieval
```bash
bash scripts/run_process_data.sh \
  --data_root ./data \
  --cuda_id 0 \
  --retrieve_mode owl_agent \
  --prompt_subject product_group_name \
  --qwen_model Qwen/Qwen3-4B-Instruct-2507 \
  --text_encoder Qwen/Qwen3-Embedding-4B
```

# Stage 2: Training

The recommended command is:

```bash
bash scripts/train_model.sh   \
  --data_root ./data   \
  --output_root ./output   \
  --item_prefix clip   \
  --use_knowledge 1   \
  --use_two_tower 1   \
  --cuda_id 6   \
  --batch_size 32   \
  --num_negatives 8   \
  --max_history_items 10  \
  --candidate_chunk_size 2   \
  --item_encode_chunk_size 512   \
  --prompt_negative_mode bottomk   \
  --prompt_negatives_per_positive 2   \
  --lambda_prompt_infonce 0.05   \
  --amp 1  \
  --transformer_injection True   \
  --transformer_injection_layers 3  \
  --transformer_injection_strength 0.5 
```

The model uses frozen CLIP-like features and a lightweight trainable Transformer. Knowledge prompts are injected layerwise. Prompt-level InfoNCE uses selected prompts as positives and bottom-k least similar prompts as negatives. Recommendation negatives remain unobserved candidate items for 7-day prediction only; they are not interpreted as semantic negatives.

# Stage 3: Inference

After training, run:

```bash
bash scripts/run_social_inference.sh \
  --data_root ./data \
  --output_root ./output \
  --social_output_root ./social_output/k \
  --item_prefix clip \
  --use_knowledge 1 \
  --use_two_tower 0 \
  --cuda_id 0 \
  --checkpoint ./output/clip_transformer_k/best_model.pt \
  --covid_csv ./data/external/owid-covid-data.csv \
  --covid_location World \
  --auto_generate_axis_prompts 1 \
  --qwen_model Qwen/Qwen3-4B-Instruct-2507 \
  --axis_score_space item_text \
  --text_encoder auto \
  --use_copula_calibration 0 \
  --use_google_mobility 0
```


# Stage 4: COVID-centered social analysis

## Covid Style Shift Test, based on axis

```bash
bash scripts/run_social_analysis.sh \
  --social_output_root ./social_output/k \
  --data_root ./data \
  --exp_id axis_covid_style_shift \
  --event_month 2020-03 \
  --task1_covid_end_month 2020-09 \
  --task1_bootstrap_n 500 \
  --make_figures 1
```

## Lead-lag

```bash
bash scripts/run_social_analysis.sh \
  --social_output_root ./social_output/k \
  --data_root ./data \
  --exp_id lead_lag \
  --event_month 2020-03 \
  --covid_csv ./data/external/owid-covid-data.csv \
  --covid_location World \
  --lead_lag_start_date 2019-10-01 \
  --lead_lag_end_date 2020-09-30 \
  --lead_lag_covid_transform delta7 \
  --lead_lag_roll_days 7 \
  --lead_lag_max_lag_days 28 \
  --lead_lag_placebo_days 7 \
  --lead_lag_bins 0:3,4:7,8:14,15:28 \
  --make_figures 1
```

## User Heterogeneity

```bash
bash scripts/run_social_analysis.sh \
  --social_output_root ./social_output/k \
  --data_root ./data \
  --exp_id user_heterogeneity \
  --event_month 2020-03 \
  --user_covid_end_month 2020-09 \
  --max_purchase_rows 0 \
  --user_use_cuda auto \
  --user_txn_weighting user_balanced \
  --user_run_transaction_kde 1 \
  --user_run_transaction_gmm 1 \
  --user_run_user_gmm 1 \
  --make_figures 1
```

## Channel Migration

```bash
bash scripts/run_social_analysis.sh \
  --social_output_root ./social_output/k \
  --data_root ./data \
  --exp_id channel_migration \
  --event_month 2020-03 \
  --channel_target_id 2 \
  --channel_covid_end_month 2020-09 \
  --covid_csv ./data/external/owid-covid-data.csv \
  --covid_location World \
  --channel_regression_start_date 2019-10-01 \
  --channel_regression_end_date 2020-09-30 \
  --channel_covid_transform delta7 \
  --channel_lag_bins 0:3,4:7,8:14,15:28 \
  --make_figures 1
```

