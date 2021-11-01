# Incorporating Global Information in Local Attention for Knowledge Representation Learning

Source code for our paper: [Incorporating Global Information in Local Attention for Knowledge Representation Learning](https://aclanthology.org/2021.findings-acl.115.pdf)



### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
- pytorch (version >= 1.1.0)

### Dataset
The four public benchmark datasets for link prediction experiments with their folder names are given below.

- Freebase: FB15k-237
- Nell: NELL-995
- Kinship: kinship
- UMLS: umls

### Training      
When running for first time:

        $ sh prepare.sh

Then reproducing the results in the paper by:

* **Nell**

        $ python3 main.py --get_2hop True --use_2hop True
        
* **Kinship** 
       
        $ python3 main.py --data ./data/kinship/ --output_folder ./checkpoints/kinship/out/ --lr 1e-2 --epochs_gat 4000 --epochs_conv 400 --batch_size_gat 8544 --drop_GAT 0.3 --weight_decay_conv 1e-5 --valid_invalid_ratio_conv 10 --out_channels 50 --drop_conv 0.0 --get_2hop True --use_2hop True
        
* **Other datasets**

 Parameters of other datasets are given in appendix of the paper. 


For any comments or suggestions, please contact zhhan@connect.hku.hk
