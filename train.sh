#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/unet.py 4
CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/unet_kernel_select.py 4
#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/unet_transformer_block.py 4
#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/unet_all.py 4
#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/unetpp.py 4
#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/transunet.py 4
#CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh ./my_config/swinunet.py 4
