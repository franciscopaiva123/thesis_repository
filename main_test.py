import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# 0 = test navy
# 1 = test seaItal
test = 1

if test == 0:
    with open('Portuguese_Navy_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X = []
    y = []
    exclude_cols = ['VesselID', 'utc', 'lat', 'lon']

    for enhanced_df, label in data:
        X.append(enhanced_df)
        y.append(label)
else:
    with open('SeaITall_data.pkl', 'rb') as f:
        evaluation_data = pickle.load(f)
    X = []
    y = []
    exclude_cols = ['VesselID', 'utc', 'lat', 'lon']

    for enhanced_df, label in evaluation_data:
        # Extract only relevant features
        feature_cols = [col for col in enhanced_df.columns if col not in exclude_cols]
        # Get feature sequence
        feature_sequence = enhanced_df[feature_cols].values
        X.append(feature_sequence)
        y.append(label)

X = np.array(X)
y = np.array(y)

labels = ['purseseines', 'fixedgear', 'drift', 'trawler']
# Manually create a mapping of classes to indices
class_to_index = {label: idx for idx, label in enumerate(labels)}
# Convert the labels to integer indices
integer_encoded = np.array([class_to_index[label] for label in y])
# Manually create a one-hot encoded matrix
n_classes = len(labels)
one_hot_encoded = np.zeros((len(y), n_classes))
one_hot_encoded[np.arange(len(y)), integer_encoded] = 1
y = one_hot_encoded

print(X.shape)
print(y.shape)

model = load_model("CNN_LSTM.keras")

y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)

print("True Labels:", y)
print("Predicted Labels:", y_pred)

# Map integer predictions back to string labels for better readability
predicted_labels = [list(class_to_index.keys())[idx] for idx in y_pred]
true_labels_str = [list(class_to_index.keys())[idx] for idx in integer_encoded]

y_true_indices = np.argmax(y, axis=1)
unique_labels = np.unique(np.concatenate([y_true_indices, y_pred]))
report = classification_report(
    y_true_indices,
    y_pred,
    labels=unique_labels,
    target_names=[labels[i] for i in unique_labels]
)
print(report)

accuracy = accuracy_score(y_true_indices, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_true_indices, y_pred, labels=np.arange(len(labels)))

cm_normalized = cm.astype('float') / cm.sum()

# Normalize confusion matrix for better visualization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Normalize by column sum (precision normalization)
cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

# Handle division by zero (if a column sum is zero, set it to 0)
cm_precision = np.nan_to_num(cm_precision)

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_precision, display_labels=labels)
disp.plot(cmap='viridis')  # Correct usage of viridis colormap

# Add a title
plt.title("Normalized Confusion Matrix")
plt.show()
