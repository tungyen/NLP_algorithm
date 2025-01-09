import argparse
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='Mix files')
    parser.add_argument('--file1', type=str, help='First filename')
    parser.add_argument('--file2', type=str, help='Second filename')
    parser.add_argument('--output', type=str, help='Output filename')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    with open(args.file1, 'rb') as f1:
        ls1 = f1.readlines()
    with open(args.file2, 'rb') as f2:
        ls2 = f2.readlines()

    # Shuffle
    ls = ls1 + ls2
    random.shuffle(ls)
    
    with open(args.output, 'wb') as fo:
        fo.writelines(ls)