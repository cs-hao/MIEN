# MIEN
This is the official implementations of Multi-scale interactive enhanced network (MIEN).

## Install

- Pytorch 1.0

- torchvision 0.4.0
- tqdm
- scipy
- scikit-image
- torchstat
- numpy
- imageio

## Training

Cd to 'TrainCode/code', run the following scripts to train models.

`TrainCode` <br/>
  `└──`code<br/>

​	 `└──`experiment<br/>

### x2

``` bash
CUDA_VISIBLE_DEVICES=0  python main.py  --model MIEN --save MIEN_x2 --scale 2  --patch_size 96  --chop --save_results 

# loading from breakpoint...
CUDA_VISIBLE_DEVICES=0  python main.py  --model BCAN --save MIEN_x2 --scale 2  --patch_size 96  --resume - 1 --load MIEN_x2 --chop --save_results 
```

### x3

``` bash
CUDA_VISIBLE_DEVICES=0  python main.py  --model MIEN --save MIEN_x3 --scale 3  --patch_size 144  --chop --save_results 

# loading from breakpoint...
CUDA_VISIBLE_DEVICES=0  python main.py  --model BCAN --save MIEN_x3 --scale 3  --patch_size 144  --resume - 1 --load MIEN_x3 --chop --save_results
```

### x4

```bash
CUDA_VISIBLE_DEVICES=0  python main.py  --model MIEN --save MIEN_x4 --scale 4  --patch_size 144  --chop --save_results 

# loading from breakpoint...
CUDA_VISIBLE_DEVICES=0  python main.py  --model BCAN --save MIEN_x4 --scale 4  --patch_size 144  --resume - 1 --load MIEN_x4 --chop --save_results
```

## Testing

Cd to 'TestCode/code', run the following scripts to train models.

`TestCode` <br/>
  `└──`code<br/>

​	`└──`HR<br/>

​			`└──`Set5<br/>

​					`└──`x2/x3/x4<br/>

​	`└──`LR<br/>

​			`└──`LRBI<br/>

​			`└──`Set5<br/>

​					`└──`x2/x3/x4<br/>

​	`└──`SR<br/>

​	`└──`model<br/>

### x2

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop --save MIEN  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop --save MIEN  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop --save MIEN  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop --save MIEN  --testset Urban100
///////////////////////////////////////////////////////////////////////////////////////
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2  --model MIEN --pre_train ../model/MIEN_x2.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Urban100
```

### x3
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop --save MIEN  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop --save MIEN  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop --save MIEN  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop --save MIEN  --testset Urban100
///////////////////////////////////////////////////////////////////////////////////////
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3  --model MIEN --pre_train ../model/MIEN_x3.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Urban100
```

### x4
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop --save MIEN  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop --save MIEN  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop --save MIEN  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop --save MIEN  --testset Urban100
///////////////////////////////////////////////////////////////////////////////////////
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4  --model MIEN --pre_train ../model/MIEN_x4.pt --test_only --save_results --chop  --self_ensemble --save MIENPlus  --testset Urban100
```




