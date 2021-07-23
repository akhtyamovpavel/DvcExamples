from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


import pandas as pd
from argparse import ArgumentParser
import os

import json


def parse_args():
    parser = ArgumentParser('Train model')
    parser.add_argument('--input', help='Path to train/val folder', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LogisticRegression())
    ])

    train_path = os.path.join(args.input, 'train.csv')
    val_path = os.path.join(args.input, 'val.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)


    pipeline.fit(train_df.text, train_df.source)
    
    train_pred_source = pipeline.predict(train_df.text)

    val_pred_source = pipeline.predict(val_df.text)

    train_accuracy = accuracy_score(train_pred_source, train_df.source)
    val_accuracy = accuracy_score(val_pred_source, val_df.source)
    print(train_accuracy, val_accuracy)

    with open('metrics.json', 'w') as fp:
        json.dump({
                'train': {
                    'accuracy': train_accuracy
                },
                'val': {
                    'accuracy': val_accuracy
                }
            },
        fp)

