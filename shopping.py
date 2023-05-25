import csv
import sys
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename)as file:
       evidence = []
       labels = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header row

        for row in reader:
            # Convert columns to their appropriate data types
            admin = int(row[0])
            admin_dur = float(row[1])
            info = int(row[2])
            info_dur = float(row[3])
            prod = int(row[4])
            prod_dur = float(row[5])
            bounce = float(row[6])
            exit = float(row[7])
            page = float(row[8])
            special = float(row[9])
            month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(row[10])
            os = int(row[11])
            browser = int(row[12])
            region = int(row[13])
            traffic = int(row[14])
            visitor = 1 if row[15] == 'Returning_Visitor' else 0
            weekend = 1 if row[16] == 'TRUE' else 0

            # Add to evidence and labels lists
            evidence.append([admin, admin_dur, info, info_dur, prod, prod_dur, bounce, exit, page, special, month, os, browser, region, traffic, visitor, weekend])
            labels.append(1 if row[17]=='TRUE' else 0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    knn=KNeighborsClassifier(n_neighbors=1)
    return knn.fit(evidence,labels)
    
    


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            true_positives += 1
        elif labels[i] == 0 and predictions[i] == 0:
            true_negatives += 1
        elif labels[i] == 0 and predictions[i] == 1:
            false_positives += 1
        elif labels[i] == 1 and predictions[i] == 0:
            false_negatives += 1

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity
    


if __name__ == "__main__":
    main()
