python3 -m experiments.mvp --task parity --model transformer --eval-every 200 --eval-batches 50 --patience 10 --min-delta 1e-3 --no-tqdm --seq-len 128
eval step=200 loss=0.8878 acc=0.4994 seq_len=128
eval step=400 loss=0.6974 acc=0.4916 seq_len=128
eval step=600 loss=0.7422 acc=0.4906 seq_len=128
eval step=800 loss=0.0000 acc=1.0000 seq_len=128
eval step=1000 loss=0.0000 acc=1.0000 seq_len=128
eval step=1200 loss=0.0000 acc=1.0000 seq_len=128
eval step=1400 loss=0.0000 acc=1.0000 seq_len=128
eval step=1600 loss=0.0001 acc=1.0000 seq_len=128
eval step=1800 loss=2.4604 acc=0.8409 seq_len=128
eval step=2000 loss=2.4330 acc=0.8356 seq_len=128