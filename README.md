# Incorporating Global Information in Local Attention for Knowledge Representation Learning

Source code for our paper: Incorporating Global Information in Local Attention for Knowledge Representation Learning



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

        $ python3 main.py --get_2hop True

* **Other datasets**

 Parameters of other datasets are given in appendix of the paper. 


For any comments or suggestions, please contact zhhan@connect.hku.hk