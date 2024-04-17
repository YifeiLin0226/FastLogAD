# FastLogAD #
Official Implementation of "FastLogAD: Log Anomaly Detection with Mask-Guided Pseudo Anomaly Generation and Discrimination" [read here](https://arxiv.org/abs/2404.08750v1)
## Install Requirements ##
```
pip install -r requirements.txt
```

## Process Data ##
1. Download and extract HDFS dataset from [here](https://zenodo.org/records/8196385)
2. Before processing data, set the following directories for each corresponding dataset.py under **dataprocess** directory:
   ```
   input_dir =   # i.e. .../BGL/
   output_dir = # i.e. .../BGL/output/
   ```
3. Run parsing and processing(into sequences):
    ```
    python -m dataprocess.[dataset name]
    ```
    [dataset name]: hdfs, bgl, thunderbird


## Run FastLogAD ##
1. Set the proper directories and hyperparameters in each .yaml file under **configs**
2. Run the main module with your choice of the generator variant(MLM, Random)

    i.e. MLM generator variant on HDFS:
    ```
    python -m main --config hdfs.yaml
    ```

    Random generator variant on Thunderbird:
    ```
    python -m main --config thunderbird.yaml
    ```
