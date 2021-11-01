####################################  TRAIN  ########################################
# x2
python main.py  --model BCAN --save BCAN_x2 --scale 2  --chop --save_results --patch_size 96

# x3
python main.py  --model BCAN --save BCAN_x3 --scale 3  --chop --save_results --patch_size 144

# x4
python main.py  --model BCAN --save BCAN_x4 --scale 4  --chop --save_results --patch_size 192



# CUDA_VISIBLE_DEVICES

# 训练
CUDA_VISIBLE_DEVICES = 0 python main.py  --model BCAN --save BCAN_x4 --scale 4  --patch_size 96  --chop --save_results 

CUDA_VISIBLE_DEVICES = 0 python main.py  --model BCAN --save BCAN_x4 --scale 4  --patch_size 144  --chop --save_results 

CUDA_VISIBLE_DEVICES = 0 python main.py  --model BCAN --save BCAN_x4 --scale 4  --patch_size 192  --chop --save_results 


# 从断点加载

CUDA_VISIBLE_DEVICES = 0 python main.py  --model BCAN --save BCAN_x4 --scale 4  --patch_size 192  --resume - 1 --load BCAN_x4 --chop --save_results 

