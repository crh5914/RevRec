import sys
import pandas as pd
import numpy as np
if len(sys.argv) > 2:
     dataset = sys.argv[1]
     path = sys.argv[2]
     src = path + dataset + '.csv'
     df = pd.read_csv(src,sep='_|_',engine='python',header=None)
     data = df[[0,2,6,4]]
     idx = list(range(len(data)))
     np.random.shuffle(idx)
     test_idx = idx[-int(0.1*len(data)):]
     valid_idx = idx[-int(0.2*len(data)):-int(0.1*len(data))]
     train_idx = idx[:-int(0.2*len(data))]
     train_data = data.iloc[train_idx]
     valid_data = data.iloc[valid_idx]
     test_data = data.iloc[valid_idx]
     train_data.to_csv(path+'train_'+dataset+'.txt',sep='\t',header=False,index=False)
     valid_data.to_csv(path+'tvalid_'+dataset+'.txt',sep='\t',header=False,index=False)
     test_data.to_csv(path+'test_'+dataset+'.txt',sep='\t',header=False,index=False)

