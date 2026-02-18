def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(filepath):

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} cars from {filepath}")
    return df

def split_data(features, classes, test_size=0.20, random_state=42):

    features_train, features_test, classes_train, classes_test = train_test_split(
        features, classes, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(features_train)}")
    print(f"Testing set size:  {len(features_test)}")
    return features_train, features_test, classes_train, classes_test

def learn_tree(features_train, classes_train, criterion="gini", max_depth=None):

#----Train part on trainign data
    decision_tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    decision_tree.fit(features_train, classes_train)
    return decision_tree

def visualise_tree(decision_tree, feature_names, class_names, output_path):

    plt.figure(figsize=(20, 10))
    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree - Car Style Prediction", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def print_predictions(classes_test, classes_predicted, accuracy):

    print(f"Expected classes\n{classes_test}")
    print(f"Predicted classes\n{classes_predicted}")
    print(f"Accuracy: {accuracy:.2f}")

def save_tree_cars(test_df, classes_predicted, accuracy, output_path):

    results = test_df[["Volume", "Doors", "Style"]].copy()
    results["PredictedStyle"] = classes_predicted

    accuracy_row = pd.DataFrame([{
        "Volume": "Accuracy",
        "Doors": "",
        "Style": "",
        "PredictedStyle": round(accuracy, 4)
    }])
    output = pd.concat([results, accuracy_row], ignore_index=True)
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main():

    df = load_data("AllCars.csv")

    feature_names = ["Volume", "Doors"]
    class_names = sorted(df["Style"].unique().tolist())
    print(f"Feature names: {feature_names}")
    print(f"Class names:   {class_names}")

    features = df[feature_names].to_numpy()
    classes = df["Style"].tolist()

    features_train, features_test, classes_train, classes_test = split_data(
        features, classes, test_size=0.20, random_state=42
    )

    _, test_df = train_test_split(df, test_size=0.20, random_state=42)

    decision_tree = learn_tree(features_train, classes_train, criterion="entropy", max_depth=5)

    visualise_tree(decision_tree, feature_names, class_names, "TreeCars.png")

    print("\nTest on training data")
    classes_predicted_train = decision_tree.predict(features_train).tolist()
    accuracy_train = accuracy_score(classes_train, classes_predicted_train)
    print_predictions(classes_train, classes_predicted_train, accuracy_train)

#----Test on testing data
    print("\nTest on testing data")
    classes_predicted = decision_tree.predict(features_test).tolist()
    accuracy = accuracy_score(classes_test, classes_predicted)
    print_predictions(classes_test, classes_predicted, accuracy)

#----Save TreeCars.csv
    save_tree_cars(test_df, classes_predicted, accuracy, "TreeCars.csv")
#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------
