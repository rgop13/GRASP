python train.py \
  --config_file GRASP-base \
  --save_dirpath checkpoint/grasp_base \
  --data_dir data/datav1 \
  --process_name GRASP_base \
  --mode train

python train.py \
  --config_file GRASP-base \
  --save_dirpath checkpoint/grasp_base \
  --data_dir data/datav1 \
  --process_name GRASP_base \
  --mode evaluation
