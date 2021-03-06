import pandas as pd
from pathlib import Path
import os
import argparse

def combine_datasets(data_dir, class_encoding='subdirectory'):
    # Search datasets folder recursively for jpg/jpeg files
    paths = Path(data_dir).rglob('*.jp*')
    df = pd.DataFrame({ 'filepath': paths })

    # Make file path absolute
    to_abs = lambda path: os.path.abspath(path)
    df['filepath'] = df['filepath'].map(to_abs)

    # Exclude unwanted folders
    to_exclude = df['filepath'].str.contains('asl_alphabet_test') | \
                df['filepath'].str.contains('Pre-Processed')
    df = df[~to_exclude].reset_index(drop=True)

    # Extract filename from path using basename
    df['filename'] = df['filepath'].map(os.path.basename)

    # Define class as last subdirectory
    if class_encoding == 'subdirectory':
        df['dataset'] = df['filepath'].apply(os.path.relpath, args=(data_dir,))
        df['dataset'] = df['dataset'].str.split('/').str[0]
        df['class'] = df['filepath'].str.split('/').str[-2]
    else: # could also use first filename character
        df['class'] = df['filename'].str[0]

    # Only keep alphabetic classes
    df = df[df['class'].str.match(r'^[a-zA-Z]$')]

    # Normalize to make all class names uppercase alphabet
    df['class'] = df['class'].str.upper()

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine datasets")
    parser.add_argument('data_dir', metavar='d')
    parser.add_argument('--class_encoding', default='subdirectory')
    args = parser.parse_args()

    print(f'Combining datasets from {args.data_dir}...')
    df = combine_datasets(args.data_dir, class_encoding=args.class_encoding)

    # Save combined dataset as .csv
    indexfile = os.path.join(args.data_dir, 'data.csv')
    df.to_csv(indexfile, index=False)
    print(f'Datasets combined ✓ [index at {indexfile}]')
