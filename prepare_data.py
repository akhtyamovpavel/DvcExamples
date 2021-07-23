import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Train dataset parser')
    parser.add_argument('--input', help='Path to DataFrame', required=True)
    parser.add_argument('--output', help='Path to output frame', required=True)
    return parser.parse_args()


MAPPING_DICT = {'mchsgov': 0, 'mil': 1, 'mospolice': 2, 'russianpost': 3}


if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(args.input)

    df.dropna(axis=0, subset=['text'], inplace=True)
    df['source'] = df['source'].map(MAPPING_DICT)

    df.to_csv(args.output, index=None)
