BASE=/mnt/data1
DATA=$BASE/datasets
LOG=$BASE/logs
LFW=$DATA/face/lfw
MODEL=$BASE/models/face
python facenet_train_classifier.py --logs_base_dir $LOG --models_base_dir $MODEL --data_dir $LFW/lfw_mtcnnpy_160_part_train --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir $LFW/lfw_mtcnnpy_160 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file ../data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-4 --center_loss_alfa 0.9
