
# Import libraries
import matplotlib
matplotlib.use('Agg')  # Required for GitHub Actions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score
)

sns.set(style='white')


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

dataset = pd.read_csv('iris.csv')

# Clean column names
dataset.columns = [
    colname.replace(' (cm)', '').strip().replace(' ', '_')
    for colname in dataset.columns
]

features_names = dataset.columns.tolist()[:4]


# --------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------

dataset['sepal_length_width_ratio'] = (
    dataset['sepal_length'] / dataset['sepal_width']
)

dataset['petal_length_width_ratio'] = (
    dataset['petal_length'] / dataset['petal_width']
)

# Select features and target
dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]


# --------------------------------------------------
# 3. Split Dataset
# --------------------------------------------------

train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset['target']
)

X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')


# --------------------------------------------------
# 4. Logistic Regression Model
# --------------------------------------------------

logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=100
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(y_test, predictions_lr, average='micro')
prec_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100


# --------------------------------------------------
# 5. Random Forest Classifier
# --------------------------------------------------

rf_reg = RandomForestClassifier(random_state=44)

rf_reg.fit(X_train, y_train)

predictions_rf = rf_reg.predict(X_test)

cm_rf = confusion_matrix(y_test, predictions_rf)

f1_rf = f1_score(y_test, predictions_rf, average='micro')
prec_rf = precision_score(y_test, predictions_rf, average='micro')
recall_rf = recall_score(y_test, predictions_rf, average='micro')

train_acc_rf = rf_reg.score(X_train, y_train) * 100
test_acc_rf = rf_reg.score(X_test, y_test) * 100


# --------------------------------------------------
# 6. Confusion Matrix Plotting Function
# --------------------------------------------------

def plot_cm(cm, target_name, title, filename, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(10, 7))

    display_cm = cm.astype('float')

    if normalize:
        display_cm = display_cm / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(display_cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_name))

    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    thresh = display_cm.max() / 2

    for i, j in itertools.product(
        range(display_cm.shape[0]),
        range(display_cm.shape[1])
    ):
        if normalize:
            text_value = "{:0.2f}".format(display_cm[i, j])
        else:
            text_value = "{:,}".format(cm[i, j])

        plt.text(
            j, i, text_value,
            horizontalalignment="center",
            color="white" if display_cm[i, j] > thresh else "black"
        )

    plt.tight_layout()

    plt.ylabel('True Label')
    plt.xlabel(
        'Predicted Label\nAccuracy={:0.2f}; Misclassification={:0.2f}'.format(
            accuracy * 100, misclass * 100
        )
    )

    plt.savefig(filename, dpi=120)
    plt.close()


# Generate confusion matrices
target_name = np.array([
    'setosa',
    'versicolor',
    'virginica'
])

plot_cm(
    cm_lr,
    target_name,
    title="Confusion Matrix - Logistic Regression",
    filename="ConfusionMatrix_LR.png"
)

plot_cm(
    cm_rf,
    target_name,
    title="Confusion Matrix - Random Forest",
    filename="ConfusionMatrix_RF.png"
)


# --------------------------------------------------
# 7. Random Forest Feature Importance
# --------------------------------------------------

importances = rf_reg.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame(
    list(zip(labels, importances)),
    columns=['feature', 'importance']
)

features = feature_df.sort_values(
    by='importance',
    ascending=False
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='importance',
    y='feature',
    data=features
)

ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Random Forest Feature Importances', fontsize=12)

plt.tight_layout()
plt.savefig('FeatureImportance.png', dpi=120)
plt.close()


# --------------------------------------------------
# 8. Save Model Evaluation Scores
# --------------------------------------------------

with open('scores.txt', 'w') as score:

    score.write("========== RANDOM FOREST ==========\n")

    score.write(
        "Training Accuracy: %.2f%%\n" % train_acc_rf
    )
    score.write(
        "Testing Accuracy: %.2f%%\n" % test_acc_rf
    )
    score.write(
        "F1 Score: %.2f%%\n" % (f1_rf * 100)
    )
    score.write(
        "Recall Score: %.2f%%\n" % (recall_rf * 100)
    )
    score.write(
        "Precision Score: %.2f%%\n" % (prec_rf * 100)
    )

    score.write("\n========== LOGISTIC REGRESSION ==========\n")

    score.write(
        "Training Accuracy: %.2f%%\n" % train_acc_lr
    )
    score.write(
        "Testing Accuracy: %.2f%%\n" % test_acc_lr
    )
    score.write(
        "F1 Score: %.2f%%\n" % (f1_lr * 100)
    )
    score.write(
        "Recall Score: %.2f%%\n" % (recall_lr * 100)
    )
    score.write(
        "Precision Score: %.2f%%\n" % (prec_lr * 100)
    )


# --------------------------------------------------
# 9. Print Results
# --------------------------------------------------

print("\n========== RANDOM FOREST ==========")

print(f"Training Accuracy: {train_acc_rf:.2f}%")
print(f"Testing Accuracy: {test_acc_rf:.2f}%")
print(f"F1 Score: {f1_rf * 100:.2f}%")
print(f"Recall Score: {recall_rf * 100:.2f}%")
print(f"Precision Score: {prec_rf * 100:.2f}%")


print("\n========== LOGISTIC REGRESSION ==========")

print(f"Training Accuracy: {train_acc_lr:.2f}%")
print(f"Testing Accuracy: {test_acc_lr:.2f}%")
print(f"F1 Score: {f1_lr * 100:.2f}%")
print(f"Recall Score: {recall_lr * 100:.2f}%")
print(f"Precision Score: {prec_lr * 100:.2f}%")


print("\nAll model training and evaluation tasks completed successfully!")

print("\nGenerated Files:")
print("1. ConfusionMatrix_LR.png")
print("2. ConfusionMatrix_RF.png")
print("3. FeatureImportance.png")
print("4. scores.txt")
