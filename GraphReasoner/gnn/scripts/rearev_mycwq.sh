
###ReaRev+SBERT training
# python main.py ReaRev --is_eval --load_experiment relbert-full_cwq-rearev-final.ckpt --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-sbert --data_folder data/CWQ/ --num_epoch 100 --warmup_epoch 80

###ReaRev+LMSR training
# python main.py ReaRev  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2  \
# --lm relbert --num_iter 2 --num_ins 3 --num_gnn 3  --name cwq \
# --experiment_name prn_cwq-rearev-lmsr  --data_folder data/CWQ/ --num_epoch 100 #--warmup_epoch 80


###Evaluate mycwq
/home/zhengu9/miniconda3/envs/gnn-rag/bin/python -m gnn.main ReaRev --entity_dim 50 --num_epoch 2 --batch_size 1 --eval_every 1 --data_folder data/myset/ --lm sbert --num_iter 1 --num_ins 1 --num_gnn 1 --relation_word_emb False --experiment_name ReaRev_myset --name myset