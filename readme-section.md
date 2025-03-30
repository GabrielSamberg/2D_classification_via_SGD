## Command Line Usage

This project provides a flexible command line interface through `your_script.py` that allows you to customize various training parameters.

### Basic Usage

```bash
python your_script.py [--data_path DATA_PATH] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] 
                      [--batch_size BATCH_SIZE] [--save_dir SAVE_DIR]
```

### Parameters

- `--data_path`: Path to your .mrc format dataset
  - Optional: If not provided, the script will use default settings
  - Example: `--data_path ./data/my_dataset.mrc`

- `--epochs`: Number of training epochs
  - Optional: Defaults to 10 if not specified
  - Example: `--epochs 20`

- `--learning_rate`: Learning rate for the optimizer
  - Optional: Defaults to 0.001 if not specified
  - Example: `--learning_rate 0.0005`

- `--batch_size`: Batch size for training
  - Optional: Defaults to 32 if not specified
  - Example: `--batch_size 64`

- `--save_dir`: Directory to save model checkpoints
  - Optional: Defaults to './checkpoints' if not specified
  - Example: `--save_dir ./my_models`

### Examples

**Train with all default parameters:**
```bash
python your_script.py
```

**Train with custom dataset:**
```bash
python your_script.py --data_path ./data/custom_dataset.mrc
```

**Full customization:**
```bash
python your_script.py --data_path ./data/custom_dataset.mrc --epochs 50 --learning_rate 0.0001 --batch_size 128 --save_dir ./trained_models
```

**Get help information:**
```bash
python your_script.py --help
```

### Notes

- All parameters are optional and have sensible defaults
- The script will create the save directory if it doesn't exist
- When providing a dataset, ensure it's in the .mrc format
