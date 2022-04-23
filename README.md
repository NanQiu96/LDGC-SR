# LDGC-SR


## Paper data and code

We have implemented our methods in Pytorch, and this is the source code for the Paper: "LDGC-SR: Integrating Long-range Dependencies and Global Context Information for Session-based Recommendation".

##### Datasets

You can download the four datasets (Diginetica, Tmall, Nowplaying and Retailrocket) that we used in this paper from https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1 and https://www.kaggle.com/retailrocket/ecommerce-dataset. After downloaded all datasets, please put them in the folder 'datasets/'.



## Environment

- Python 3.7
- PyTorch 1.5.0
- tqdm



## Usage

You need to run the data processing file first to preprocess the corresponding data.

For example: 

```
python preprocess.py --dataset diginetica
python process_tmall.py --dataset Tmall
python process_nowplaying.py --dataset Nowplaying
python preprocess.py --dataset retailrocket
```



Next, you need to calculate the neighbors of each item in all sessions.

For example:

```
python build_graph.py --dataset diginetica --sample_num 12

usage: build_graph.py [-h] [--dataset DATASET] [--sample_num SAMPLE_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     diginetica/Tmall/Nowplaying
  --sample_num SAMPLE_NUM
```



Finally, you can run the file 'main.py' to train and evaluate the model:

For example: 

```
# Diginetica
python main.py --dataset diginetica --dropout_global 0.8 --lambda_ 0.1 --norm --scale
# Tmall
python main.py --dataset Tmall --dropout_global 0.6 --lambda_ 0.1 --norm --scale
# Nowplaying
python main.py --dataset Nowplaying --dropout_global 0.4 --lambda_ 0.1 --norm --scale
# Retailrocket
python main.py --dataset retailrocket --dropout_global 0.6 --lambda_ 0.1 --norm --scale

usage: main.py [-h] [--dataset DATASET] [--hiddenSize HIDDENSIZE]
               [--epoch EPOCH] [--activate ACTIVATE]
               [--n_sample_all N_SAMPLE_ALL] [--n_sample N_SAMPLE]
               [--batch_size BATCH_SIZE] [--lr LR] [--lr_dc LR_DC]
               [--lr_dc_step LR_DC_STEP] [--l2 L2] [--n_iter N_ITER]
               [--step STEP] [--dropout_gcn DROPOUT_GCN]
               [--dropout_global DROPOUT_GLOBAL] [--validation]
               [--valid_portion VALID_PORTION] [--alpha ALPHA]
               [--patience PATIENCE] [--norm] [--scale] [--tau TAU]
               [--lambda_ LAMBDA_] [--last_len LAST_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     diginetica/Nowplaying/Tmall
  --hiddenSize HIDDENSIZE
  --epoch EPOCH
  --activate ACTIVATE
  --n_sample_all N_SAMPLE_ALL
  --n_sample N_SAMPLE
  --batch_size BATCH_SIZE
  --lr LR               learning rate.
  --lr_dc LR_DC         learning rate decay.
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate
                        decay.
  --l2 L2               l2 penalty
  --n_iter N_ITER
  --step STEP           star graph propogation steps
  --dropout_gcn DROPOUT_GCN
                        Dropout rate.
  --dropout_global DROPOUT_GLOBAL
                        Dropout rate.
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion
  --alpha ALPHA         Alpha for the leaky_relu.
  --patience PATIENCE
  --norm                adapt NISER, l2 norm over item and session embedding
  --scale               scaling factor sigma
  --tau TAU             scale factor of the scores.
  --lambda_ LAMBDA_     weight of short-term memory.
  --last_len LAST_LEN   the number of last items
```

