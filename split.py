from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='run extract script')
    parser.add_argument('--dataset',type=str,default='baby',help='specify the dataset')
    parser.add_argument('--ratio',type=float,default=0.2,help='test ratio')
    return parser.parse_args()

def split(prefix,ratio):
    #users,items,reviews,ratings,timestamps = [],[],[],[],[]
    data = []
    file = prefix + '.csv'
    with open(file,'r') as fp:
        for line in fp:
            vals = line.strip().split('_|_')
            data.append(vals)
    train_data,test_data = train_test_split(data,test_size=ratio)
    with open(prefix+'_train.csv','w') as fp:
        for vals in train_data:
            line = ','.join(vals)
            fp.write(line+'\n')
    with open(prefix+'_test.csv','w') as fp:
        for vals in test_data:
            line = ','.join(vals)
            fp.write(line+'\n')
def main():
    args = parse_args()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    split(prefix,args.ratio)
if __name__ == '__main__':
    main()
