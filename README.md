## Command Line Usage

This project provides a flexible command line interface through `model.py` that allows you to customize various training parameters.

### Basic Usage

```bash
python model.py [--data_path DATA_PATH] [--epochs EPOCHS] [--learning_rate_theta LEARNING_RATE_THETA] [--learning_rate_sigma LEARNING_RATE_SIGMA] 
                      [ --train_pre_model TRAIN_PRE_MODEL] [ --pre_epochs PRE_EPOCHS] [--generate_dataset GENERATE_DATASET]
```

### Parameters

- `--data_path`: Path string to your .mrcs format dataset
  - Optional: If not provided, the script will generate our defaul dataset
  - Example: `--data_path './data/my_dataset.mrc'`

- `--epochs`: Number of training epochs for the main model
  - Optional: Defaults to 10 if not specified
  - Example: `--epochs 20`

- `--learning_rate_theta`: Learning rate for the theta parameter of the optimizer
  - Optional: Defaults to 0.01 if not specified
  - Example: `--learning_rate_theta 0.0005`

- `--learning_rate_sigma`: Learning rate for the sigma parameter of the optimizer
  - Optional: Defaults to 0.001 if not specified
  - Example: `--learning_rate_sigma 0.0005`

- `--train_pre_model`: Decide if to run a pre-training proccess that will initialize the model with a better starting point
  - Optional: Defaults to True if not specified
  - Example: `--train_pre_model False`

- ` --pre_epochs`: Number of training epochs for the pre-model 
  - Optional: Defaults to 7 if not specified
  - Example: `--pre_epochs 11`
 
- ` --generate_dataset`: If no dataset path provided, generates default dataset 
  - Optional: Defaults to False if not specified
  - Example: `--generate_datasets True`

### Examples

**Train with all default parameters:**
```bash
python model.py
```

**Train with custom dataset:**
```bash
python model.py --data_path './data/custom_mrcs_dataset'
```

**Full customization:**
```bash
python model.py --data_path './data/custom_mrcs_dataset' --epochs 50 --learning_rate_theta 0.03 --learning_rate_sigma 0.0007 --pre_epochs 11 
```

**Get help information:**
```bash
python model.py --help
```

### Notes

- All parameters are optional and have sensible defaults
- When providing a dataset, ensure it is stored in a file in which all the datasamples are in the .mrcs format
- Note that at the moment the code supports only datasets of exactly 100 data samples where each data sample is a 256x256 image
