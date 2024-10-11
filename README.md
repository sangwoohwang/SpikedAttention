# SpikedAttention: Training-Free and Fully Spike-Driven Transformer-to-SNN Conversion with Winner-Oriented Spike Shift for Softmax Operation
## What is?
This is simple test code for Spiked-Attention which accepted to Neurips 2024.
The code is designed by PyTorch Image Models(TIMM), Huggingface Hub and SpikingJelly framework.
Beacuse of limitation of supplement and annonymity, one of pre-trained ANN(swin-tiny without ReLU) is uploaded on below google drive. 

link for pretrained Swin-Transformer: https://drive.google.com/file/d/1RJH_tdjKLoUHRg0G25ccr0hG8oEQ_lmz/view?usp=sharing

link for pretrained pre-trained MA-BERT for SST-2: https://drive.google.com/file/d/1A8bdmnU77eh-mDd_Zb0-RryR0zxvBAKH/view 
** You should unzip the file 


In this code, you need to download pre-trained model on google drive.
First, the code will run ANN(swin_tiny_patch4_window7_224 or ma-bert) for scaling threshold(or weight normalization) and searching base.
Then, pre-trained parameter will converted to SNN and run it.(inferece_##.py)

## Environments
To install Environments:

```
./environ.sh
```


# How to RUN (Swin-to-SpikedAttention)


Run
```
torchrun --nproc_per_node "num_of_gpu" inference_swin.py "data_path of ImageNet" --model swin_tiny_patch4_window7_224 --batch-size "batch_size" --resume "path of pre-trained ANN" --base "base B" --timestep "number of timestep"
```

In result, you can see result (accuracy/energy) as "Swin-SpikedAttention.log"

Hyper Parameter:
1. Number of Timestep (-t)   (Default: 40)
2. Base B (1,2] *if B=2, it is binary coding (Default: 1.15)


For example, run
```
torchrun --nproc_per_node 4 inference_swin.py /workspace/dataset/imagenet --model swin_tiny_patch4_window7_224 --batch-size 64 --resume model_best.pth.tar --base 1.15 --timestep 40
```

For using pre-fixed paramter, run (SIMPLE Version)
```
torchrun --nproc_per_node 4 inference_swin.py /workspace/dataset/imagenet --batch-size 64 --resume model_best.pth.tar 
```
Note ImageNet data path and pre-trained model must be included.

# How to RUN (MABERT-to-SpikedAttention) 
** only pre-trained model (MA-BERT) for SST-2 provided
Run
```
torchrun --nproc_per_node 1 inference_glue.py --task "selected task" --batch-size "batch_size" --pretrained_file "path of pre-trained ANN(file name)" --base "base B" --timestep "number of timestep"
```
In result, you can see result (accuracy/energy) as "MABERT-SpikedAttention.log"

For example, run
```
torchrun --nproc_per_node 1  inference_glue.py --task sst2 --pretrained_file sst2-392-sst2 --batch_size 16  --base 1.4 --timestep 16
```

If you use RTX 4000 Series, you need to Run as
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" torchrun --nproc_per_node 1  inference_glue.py --pretrained_file sst2-392-sst2 --batch_size 16  --base 1.4 --timestep 16
```

Note Multi-GPU not supported on MABERT-to-SpikedAttention.
Note ImageNet pre-trained model must be included.

