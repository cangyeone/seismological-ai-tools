from base import WriteH5
import argparse 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='制作H5数据') 
    parser.add_argument("-i", "--index", type=str, help="索引位置")
    parser.add_argument("-o", "--output", type=str, help="输出文件位置") 
    parser.add_argument("-c", "--ctlg", type=str, help="震相目录位置") 
    parser.add_argument("-s", "--station", type=str, help="台站位置信息") 
    parser.add_argument("-r", "--range", type=float, default=500.0, help="截取长度") 
    parser.add_argument("-t", "--nthread", type=int, default=1, help="处理线程数") 
    args = parser.parse_args() 
    makeh5 = WriteH5(args)
