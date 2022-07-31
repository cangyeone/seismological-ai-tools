import os 
import argparse 
def make_h5(args):
    #os.environ['LIBMSEED_LEAPSECOND_FILE']="utils/leap-seconds.list"
    count = 0
    for root, dirs, files in os.walk(args.root):
        for f in files:
            if f.endswith(".mseed"):
                path = os.path.join(root, f)
                os.system(f"utils/mseedidx -sqlite {args.out} {path}")
                count += 1 
                if count %100 ==0:
                    print("已完成", count, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='制作索引') 
    parser.add_argument("-r", "--root", type=str, help="搜索路径")
    parser.add_argument("-o", "--out", type=str, help="文件位置") 
    args = parser.parse_args() 
    make_h5(args)