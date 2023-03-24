from Utils import AnnotatedEmails, AnnotatedEmail
from Utils import denotation_types
from features import mail2features
import pandas as pd
import numpy as np
import pycrfsuite
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


def flatten(lst):
    return [l for sub in lst for l in sub]


def to_array(lst, cols=None):
    df = pd.DataFrame(lst, columns=cols)
    df.fillna(0, inplace=True)
    if cols is not None:
        return np.nan_to_num(df[cols].values)
    return df.columns, df.values


if __name__ == '__main__':

    zones = 2
    
    emails = AnnotatedEmails(train_df, mail2features, perturbation=0.0)
    print('loaded mails')

    X_train, X_test, X_eval = emails.features
    print('loaded features')
    le = LabelEncoder()
    le.fit(AnnotatedEmail.zone_labels(zones))
    print(le.classes_)
    if zones == 2:
        y_train, y_test, y_eval = emails.two_zones_labels
    else:
        y_train, y_test, y_eval = emails.two_zones_labels
    class_weights = compute_class_weight(class_weight='balanced', classes=le.classes_, y=flatten(y_train))
    print('loaded labels')
