from argparse import ArgumentParser
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml


def parse_args():
    parser = ArgumentParser('Train test split')
    parser.add_argument('--input', help='Path to train dataset', required=True)
    parser.add_argument('--output_folder', required=True)
    return parser.parse_args()


def read_config():
    with open('params.yaml', 'r') as fp:
        return yaml.safe_load(fp)['split']


if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(args.input)

    params = read_config()

    train_df, val_df = train_test_split(df, test_size=params['test_size'], random_state=params['seed'])

    os.makedirs(args.output_folder, exist_ok=True)

    train_df.to_csv(os.path.join(args.output_folder, 'train.csv'), index=None)
    val_df.to_csv(os.path.join(args.output_folder, 'val.csv'), index=None)

