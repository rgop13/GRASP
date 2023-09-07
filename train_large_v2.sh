python train.py \
  --config_file GRASP-large \
  --save_dirpath checkpoint/grasp_largev2 \
  --data_dir data/datav2 \
  --process_name GRASP_large \
  --mode train

python train.py \
  --config_file GRASP-large \
  --save_dirpath checkpoint/grasp_largev2 \
  --data_dir data/datav2 \
  --process_name GRASP_large \
  --mode evaluation
