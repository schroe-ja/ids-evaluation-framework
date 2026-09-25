# How to create new Examples

```
run_config/             # All files needed to evaluate
├── <config>.yml        # Example config yml
├── raw_data/           # All datasets for evaluation
└── <dataset_name>/     # Example dataset structure 
    ├── <dataset>.csv   # Example dataset CSV
├── plugin_ids/         # All Python scripts to be evaluated in a given yml
└── <ml_model>.py       # Example Python script
```


# Current Examples

| IDS Name | Reference |
| -------- | --------- |
| Decision Tree | Baseline Model |
| DNN | [[1]](https://ieeexplore.ieee.org/document/8494096) |
| Kitsune | [[2]](https://arxiv.org/abs/1802.09089) (Pre-Print) |
| Tree Based IDS | [[3]](https://ieeexplore.ieee.org/document/9013892) |
| Apollon MAB IDS | [[4]](https://www.sciencedirect.com/science/article/pii/S016740482300456X) | 

# Cross-Dataset Evaluation Datasets

| Dataset Name | Reference |
| ------------ | --------- |
| CICIDS2017   | [[5]](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CIC-IoT-23   | [[6]](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| CIC-IoT-DataSense-2025 | [[7]](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html) |
| CIC-IoT-DIAD-2024 (Flow) | [[8]](https://www.unb.ca/cic/datasets/iot-diad-2024.html) |
