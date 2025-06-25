This file explains how to execute the various python files.

4 python files are availabel to train 4 different models:

1.main_train_1DCNN_binary.py and main_train_CNN_LSTM_binary.py

Both these files train a model to classify between fishing and non-fishing activity.

Before running the file two steps must be carried out:

First access the "preprocess.py" file and make sure that the 150st line: "df = df[(df['is_fishing'] == 1.0) | (df['is_fishing'] == 0)]" is NOT commented. And that the 151st line "df = df[(df['is_fishing'] == 1.0)]" IS commented. We want both signals that are fishing (1.0) and not fishing (0.0).

Secondly the correct path in the "main_train_1DCNN_binary.py" must be configured. Acess line 32 and define the filepath for the wanted fishing art file. For example "file_path="DATASETS/fixed_gear.csv"" *

Now run the file and the model will be trained. Model architecture can be altered by changed the contents of line 517-533.

2.main_train_1dCNN_multiclass.py and main_train_CNN_LSTM_multiclass.py

Both these files train a model to classify between various fishing arts.

Before running the file two steps must be carried out:

First access the "preprocess.py" file and make sure that the 151st line: "df = df[(df['is_fishing'] == 1.0)]" is NOT commented. And that the 150st line "df = df[(df['is_fishing'] == 1.0) | (df['is_fishing'] == 0)] IS commented. We want only signals that are fishing (1.0).

Secondly the correct paths of all the vessels signals must be configured. Acess line 30-34 and define the filepaths for all the fishing arts. For example "file_path_1 = "DATASETS/purse_seines.csv" *

Now run the file and the model will be trained. Model architecture can be altered by changed the contents of line 611-627.


3. How to evaluate the trained models:

3.1 Xsealance-SeaItall Dataset

After training the model a .keras file will be created. 

Access the main_test.py file and in line 11 set the test flag as 1.

In the same file add the model .keras file in line 55 for example "model = load_model("CNN.keras")"

Run the file.

3.2 Portuguese Navy Dataset


After training the model a .keras file will be created.

Access the main_test.py file and in line 11 set the test flag as 0.

In the same file add the model .keras file in line 55 for example "model = load_model("CNN.keras")"

Run the file.



* In order to evaluate the models against both datasets one modification must be made:

Both evaluation datasets don´t have information regarding the "troller" fishing art. As such the models must be trained without this particular fishing art.

To train the models without this fishing art the folowing lines must be commented out from the main_train_1dCNN_multiclass.py and main_train_CNN_LSTM_multiclass.py files:

dataset_clean_4 = feature_engineer_by_vessel(dataset_clean_4) # to evaluate the models comment before training
dataset_normalized_4 = normalize(dataset_clean_4) # to evaluate the models comment before training
print("Normalized data 4", len(dataset_normalized_4)) # to evaluate the models comment before training

segmented_data_4 = segment_vessel_data(dataset_normalized_4, label="troller") # to evaluate the models comment before training
print("Segmented data 4:", len(segmented_data_4)) # to evaluate the models comment before training

And these lines: 

segmented_data_all = segmented_data_1 + segmented_data_2 + segmented_data_3 + segmented_data_4 + segmented_data_5 - line 531

labels = ['purseseines', 'fixedgear', 'drift', 'trawler','troller'] - line 553

Must be changed to the folowing lines:


segmented_data_all = segmented_data_1 + segmented_data_2 + segmented_data_3 + segmented_data_5

labels = ['purseseines', 'fixedgear', 'drift', 'trawler']


Final Note:

All the model training files take around 10-15 minutes to run.

Before running the python files the Datasets must be downloaded from the following the GFW website: [GFW Dataset](https://globalfishingwatch.org/data-download/datasets/public-training-data-v1) 

A free account must be setup. The name of the dataset is "Anonymized AIS training data".



