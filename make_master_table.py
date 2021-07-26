import pickle
import pandas as pd
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--nwsi', type=int, help='number of the WSI. Will determine its class')
    parser.add_argument('--problem', type=str, help='classification prolem to solve')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dic = []
    for i in range(args.nwsi):
        dic.append({'ID': f'wsi_{i}', 'target':'{}'.format(i%2)})
    table = pd.DataFrame(dic)
    table.to_csv('table_data.csv', index=False)

    
if __name__ == '__main__':
    main()

