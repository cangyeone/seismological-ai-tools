
import h5py 
import datetime 
import pickle 
import tqdm 
from base import TestH5 
import argparse 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='测试H5数据') 
    parser.add_argument("-c", "--ctlg", type=str, help="震相数据") 
    parser.add_argument("-i", "--input", type=str, help="H5数据") 
    parser.add_argument("-o", "--output", type=str, help="统计数据") 
    args = parser.parse_args()
    #makeh5 = WriteH5()
    #makeh5.outputh5("2014")
    #datatool = ReadH5("2019")
    #datatool.plot()
    testh5 = TestH5(args.ctlg) 
    testh5.test(args.input, args.output)