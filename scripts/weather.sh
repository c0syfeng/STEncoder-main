model=STEncoderS

pred_len=96

python run.py \
  --STEncoder \
  --STEncoder_lambda 0.001 \
  --STEncoder_multiscales 336 \
  --STEncoder_wnorm Mean \
  --batch_size 64 \
  --c_out 1 \
  --checkpoints ./checkpoints/ \
  --d_ff 4 \
  --d_layers 1 \
  --d_model 4 \
  --data weather \
  --data_path weather.csv \
  --dec_in 7 \
  --des Exp \
  --e_layers 1 \
  --embed timeF \
  --enc_in 1 \
  --factor 1 \
  --features S \
  --freq h \
  --gpu 3 \
  --is_training 1 \
  --itr 5 \
  --label_len 48 \
  --learning_rate 0.0003 \
  --loss MSE \
  --lradj type1 \
  --model $model \
  --moving_avg 25 \
  --n_heads 8 \
  --num_kernels 6 \
  --num_workers 2 \
  --p_hidden_layers 2 \
  --patience 3 \
  --pred_len $pred_len \
  --root_path ./dataset/weather \
  --seasonal_patterns Monthly \
  --seq_len 336 \
  --target OT \
  --task_name long_term_forecast \
  --top_k 5 \
  --train_epochs 10 \
  --train_ratio 0.6 \
  --dd_model 128 \
  --dd_ff 128 \
  --ee_layers 2

pred_len=720

python run.py \
  --STEncoder \
  --STEncoder_lambda 1.0 \
  --STEncoder_multiscales 720 \
  --STEncoder_wnorm ReVIN \
  --batch_size 64 \
  --c_out 1 \
  --checkpoints ./checkpoints/ \
  --d_ff 4 \
  --d_layers 1 \
  --d_model 4 \
  --data weather \
  --data_path weather.csv \
  --dec_in 7 \
  --des Exp \
  --e_layers 1 \
  --embed timeF \
  --enc_in 1 \
  --factor 1 \
  --features S \
  --freq h \
  --gpu 3 \
  --is_training 1 \
  --itr 5 \
  --label_len 48 \
  --learning_rate 0.0001 \
  --loss MSE \
  --lradj fixed \
  --model $model \
  --moving_avg 25 \
  --n_heads 8 \
  --num_kernels 6 \
  --num_workers 2 \
  --p_hidden_layers 2 \
  --patience 3 \
  --pred_len $pred_len \
  --root_path ./dataset/weather \
  --seasonal_patterns Monthly \
  --seq_len 336 \
  --target OT \
  --task_name long_term_forecast \
  --top_k 5 \
  --train_epochs 50 \
  --train_ratio 0.6 \
  --dd_model 128 \
  --dd_ff 128 \
  --ee_layers 2

pred_len=1440

python run.py \
  --STEncoder \
  --STEncoder_lambda 0.1 \
  --STEncoder_multiscales 336 \
  --STEncoder_wnorm Decomp \
  --batch_size 128 \
  --c_out 1 \
  --checkpoints ./checkpoints/ \
  --d_ff 8 \
  --d_layers 1 \
  --d_model 8 \
  --data weather \
  --data_path weather.csv \
  --dec_in 7 \
  --des Exp \
  --e_layers 2 \
  --embed timeF \
  --enc_in 1 \
  --factor 1 \
  --features S \
  --freq h \
  --is_training 1 \
  --itr 5 \
  --label_len 48 \
  --learning_rate 0.0001 \
  --loss MSE \
  --lradj fixed \
  --model $model \
  --moving_avg 25 \
  --n_heads 8 \
  --num_kernels 6 \
  --num_workers 2 \
  --p_hidden_layers 2 \
  --patience 3 \
  --pred_len $pred_len \
  --root_path ./dataset/weather \
  --seasonal_patterns Monthly \
  --seq_len 336 \
  --target OT \
  --task_name long_term_forecast \
  --top_k 5 \
  --train_epochs 50 \
  --train_ratio 0.6 \
  --dd_model 128 \
  --dd_ff 128 \
  --ee_layers 2

$pred_len=2160

python run.py \
  --STEncoder \
  --STEncoder_lambda 0.01 \
  --STEncoder_multiscales 720 \
  --STEncoder_wnorm Decomp \
  --batch_size 64 \
  --c_out 1 \
  --checkpoints ./checkpoints/ \
  --d_ff 8 \
  --d_layers 1 \
  --d_model 8 \
  --data weather \
  --data_path weather.csv \
  --dec_in 7 \
  --des Exp \
  --e_layers 2 \
  --embed timeF \
  --enc_in 1 \
  --factor 1 \
  --features S \
  --freq h \
  --is_training 1 \
  --itr 5 \
  --label_len 48 \
  --learning_rate 0.0001 \
  --loss MSE \
  --lradj fixed \
  --model $model \
  --moving_avg 25 \
  --n_heads 8 \
  --num_kernels 6 \
  --num_workers 2 \
  --p_hidden_layers 2 \
  --patience 3 \
  --pred_len $pred_len \
  --root_path ./dataset/weather \
  --seasonal_patterns Monthly \
  --seq_len 336 \
  --target OT \
  --task_name long_term_forecast \
  --top_k 5 \
  --train_epochs 50 \
  --train_ratio 0.6 \
  --dd_model 512 \
  --dd_ff 512 \
  --ee_layers 3