# HINPool Official Code

This is the official code of HINPool.

## Installation

### Open a new enviroment
```
python3 -m pip install virtualenv
python3 -m venv @envname
source @envname/bin/activate
```
### Install dgl
We are using dgl version 1.1.2 and cuda version 11.8, thus running the following script will successfully install the dgl package.
```
pip install dgl==1.1.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
```

Install other packages.
```
pip install -r requirements.txt
```

## How to Run
### Constructing HINs
```
bash generate.sh
```
This bash script constructs HINs of all datasets.
If you would like to test different performances as shown in Table 7 for NCI1 or NCI109, **change the following line** and adjust to the numbers of atoms.
```
for num in 10
```
Note that when num=0, all the atom types are used.

The constructed datasets are stored in ```dataset/$name```
### HIN appearance
Node type name: `n_{order}`
Edge type name: `{source_type}-{edge_label}-{destination_type}`

#### Example
`MUTAG`:
Two atoms with atom type 2 and 3, connected with a triple bond -> `n_B-e3-n_C`

`NCI1`: (Molecular with no bond type information)
Two atoms with atom type 4 and 5, connected with an unknown bond -> `n_D-e0-n_E`

`PROTEINS`: (Order: Helix, Sheet, Turn)
Two amino acids with SSE 'Helix' and 'Sheet', connected since their distance is smaller than Å -> `n_A-e0-n_B`

More details are in the below source files responsible for constructing HINs.
`generate.sh`
`generate.py`
`HeteroDataset.py`
`utils/utils.py`

---
### Run Main Script main.sh

#### Dataset Specification.
Line 8, 9

The `dataset` can be replaced by one of the following:
- `MUTAG`
- `NCI1`
- `NCI109`
- `PROTEINS`
- `ENZYMES`

#### Setting `num_types`:

- If the `dataset` is `MUTAG`, `PROTEINS`, or `ENZYMES`, set `num_types` to `0`.
- If the `dataset` is `NCI1` or `NCI109`, set `num_types` to the number of atoms to be tested. (Use `10` for both in the main performance table.)

#### Configurations Setting
Line 11~26 are the configurations. Each corresponding name to Table 6 and its optimal parameters are as follows:


| main.sh |    Table 6    |   MUTAG   |   NCI1    |  NCI109   |  ENZYMES  |  PROTEINS   |
|:-------:|:-------------:|:---------:|:---------:|:---------:|:---------:|:-----------:|
|  batch  |  Batch size   |    20     |    200    |    200    |    150    |     200     |
|   lr    |      LR       | 10<sup>-3 | 10<sup>-3 | 10<sup>-3 | 10<sup>-3 | 5*10<sup>-4 |
|   wd    |      WD       | 10<sup>-4 | 10<sup>-4 | 10<sup>-4 | 10<sup>-3 |  10<sup>-4  |
|  head   |     Head      |     8     |    16     |    16     |    16     |      8      |
| hidden  |    Hidden     |    128    |    32     |    32     |    32     |     32      |
| fusion  | $\phi$<sub>TR |   mean    |    max    |    max    |    max    |   att_sep   |
|  skip   | $\phi$<sub>CL |  concat   |    att    |    att    |    max    |   concat    |
|  mixed  |     Mixed     |     0     |     1     |     1     |     0     |      1      |
|   mlp   |      MLP      |     1     |     0     |     0     |     1     |      1      |
|   mul   |      Mul      |     1     |     1     |     1     |     0     |      1      |
|  norm   |     Norm      |     0     |     1     |     1     |     1     |      1      |
    
#### Start Run
After the setting is completed, you can run the script.
```
bash main.sh
```
---
### Results
The final results are saved in a csv file in `results/$dataset` .

The results shown in paper are as follows:
|| MUTAG |    NCI1   |     NCI109         | ENZYMES     | PROTEINS   |     
| :-------: | :---------: | :--------: | :--------: | :--------: | :--------: |
| AUROC   | 100.0±0.0 | 87.5±0.8 | 86.2±0.8 | 92.8±1.1 | 89.8±1.5 |
| Accuracy   |  97.7±4.9 | 80.3±1.0 | 78.7±0.6 | 79.3±4.3 |  81.2±1.9 |
---
### Ablation Studies
For each ablation study:
1. w/o Cross Layer Fusion
2. w/o TAS
3. w/o TAS - Type-Aware Top-K
4. w/o RA - Type-aware Readout Fusion

Adjust the following setting in main.sh:

| Ablation | Adjust                     |
|:--------:|:-------------------------- |
|    1     | do_cross_layer -> 1        |
|    2     | pool-ratio -> 1 1 1; mul=0 |
|    3     | pool_by -> all             |
|    4     | do_TA_readout -> 1         |
