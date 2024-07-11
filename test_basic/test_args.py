import argparse
from icecream import ic

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--dataset', '-d', help='选择哪一个数据集(1-4)进行实验')
parser.add_argument('--func', '-f', help='选择哪一种方法: (1-2)')
args = parser.parse_args()

ic(args.dataset, args.func)