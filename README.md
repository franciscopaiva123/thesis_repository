# Fishing Activity Classification — Model Training & Evaluation

This README explains how to train and evaluate different machine learning models for classifying fishing activities using AIS data.

## Available Python Files

There are **4 training scripts**, each corresponding to a different model:

---

### 1. **Binary Classification Models**

- `main_train_1DCNN_binary.py`  
- `main_train_CNN_LSTM_binary.py`  

These models classify signals as **fishing** (`1.0`) or **non-fishing** (`0.0`).

#### Before Running:

1. Open `preprocess.py` and make sure:
   - ✅ **Line 150** is **uncommented**:  
     ```python
     df = df[(df['is_fishing'] == 1.0) | (df['is_fishing'] == 0)]
     ```
   - ❌ **Line 151** is **commented**:  
     ```python
     # df = df[(df['is_fishing'] == 1.0)]
     ```

2. In the training script:
   - Go to **line 32** and set the correct dataset path, for example:
     ```python
     file_path = "DATASETS/fixed_gear.csv"
     ```

#### Running:
- Run the script to start training.
- Modify model architecture in **lines 517–533**.

---

### 2. **Multiclass Classification Models**

- `main_train_1dCNN_multiclass.py`  
- `main_train_CNN_LSTM_multiclass.py`  

These models classify between multiple fishing gears (e.g., *purse seines*, *fixed gear*, *trawlers*).

#### Before Running:

1. Open `preprocess.py` and make sure:
   - ✅ **Line 151** is **uncommented**:
     ```python
     df = df[(df['is_fishing'] == 1.0)]
     ```
   - ❌ **Line 150** is **commented**:
     ```python
     # df = df[(df['is_fishing'] == 1.0) | (df['is_fishing'] == 0)]
     ```

2. In the training script:
   - Go to **lines 30–34** and define file paths for each fishing gear. Example:
     ```python
     file_path_1 = "DATASETS/purse_seines.csv"
     file_path_2 = "DATASETS/fixed_gear.csv"
     ```

#### Running:
- Run the script to start training.
- Modify model architecture in **lines 611–627**.

---

## 3. Model Evaluation

### 3.1 XSealence–SeaItAll Dataset

1. After training, a `.keras` file will be created.
2. Open `main_test.py`:
   - Set **line 11** to:
     ```python
     test = 1
     ```
   - Load the model in **line 55**:
     ```python
     model = load_model("CNN.keras")
     ```
3. Run the script.

### 3.2 Portuguese Navy Dataset

1. After training, a `.keras` file will be created.
2. Open `main_test.py`:
   - Set **line 11** to:
     ```python
     test = 0
     ```
   - Load the model in **line 55**:
     ```python
     model = load_model("CNN.keras")
     ```
3. Run the script.

---

## ⚠️ Excluding "Troller" Class for Evaluation

The evaluation datasets do **not** include the **"troller"** class. To ensure compatibility:

1. In `main_train_1dCNN_multiclass.py` and `main_train_CNN_LSTM_multiclass.py`, **comment out**:

```python
dataset_clean_4 = feature_engineer_by_vessel(dataset_clean_4)
dataset_normalized_4 = normalize(dataset_clean_4)
print("Normalized data 4", len(dataset_normalized_4))

segmented_data_4 = segment_vessel_data(dataset_normalized_4, label="troller")
print("Segmented data 4:", len(segmented_data_4))

And these lines: 

segmented_data_all = segmented_data_1 + segmented_data_2 + segmented_data_3 + segmented_data_4 + segmented_data_5 - line 531

labels = ['purseseines', 'fixedgear', 'drift', 'trawler','troller'] - line 553

Must be changed to the folowing lines:

segmented_data_all = segmented_data_1 + segmented_data_2 + segmented_data_3 + segmented_data_5

labels = ['purseseines', 'fixedgear', 'drift', 'trawler']
---

## Final Notes

- Each training script takes approximately **10–15 minutes** to run.
- Before running any scripts, make sure to download the datasets from the official GFW website:

  📥 [Global Fishing Watch – Public Training Dataset](https://globalfishingwatch.org/data-download/datasets/public-training-data-v1)
