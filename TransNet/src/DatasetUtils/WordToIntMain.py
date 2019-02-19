'''
Created on Jan 24, 2017

@author: roseck
'''


import WordToInt
import pickle
import sys


if __name__ == '__main__':
#     infile = '/Users/roseck/Documents/RecoNNData/sl50/x_Train.txt'
#     outfile = '/Users/roseck/Documents/RecoNNData/sl50/x_INT_Train.txt'
#     
    infile = '../../data/baby/valid_baby.txt'
    outfile = '../../data/baby/valid_int_baby.txt'
    dict_file = '../../data/baby/baby_dict.pkl' #word -> id
    col = 3
    
    #read from cmd line if any
    if len(sys.argv) > 1:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        dict_file = sys.argv[3]
        col = int(sys.argv[4])
    
    print('infile = ', infile)
    print('outfile = ', outfile)
    print('dictfile = ', dict_file)
    
#     col = 3
    
    
    word_to_int_dict = pickle.load( open( dict_file, "rb" ) )
    
    print('Size of dict = ', len(word_to_int_dict))
    
    WordToInt.convert_to_int(infile, outfile, word_to_int_dict, col)
    
    print('Done.')
    