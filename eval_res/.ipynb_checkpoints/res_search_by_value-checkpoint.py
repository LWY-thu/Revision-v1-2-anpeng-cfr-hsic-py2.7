import os
import sys
import glob

def main(output_dir):
    files = glob.glob("%s/results_summary_*.txt"%output_dir)
    
    for ffile in files:
        st_idx,end_idx = -1,-1
        min_pehe, max_pehe = 10, -1
        min_pehe_idx, max_pehe_idx = -1,-1 
        with open(ffile,'r') as f:
            lines=f.readlines()
            for idx in range(len(lines)):
                if 'Test' in lines[idx]:
                    st_idx = idx + 3
                    break
            for idx in range(st_idx, len(lines)):
                if lines[idx]=="\n":
                    end_idx=idx
                    break
                values = [p.strip() for p in lines[idx].split('|')]
                if float(values[1][:5])<min_pehe:
                    min_pehe = float(values[1][:5])
                    min_pehe_idx = int(values[0])
                if float(values[1][:5])>max_pehe:
                    max_pehe = float(values[1][:5])
                    max_pehe_idx = int(values[0])
            print('\nfile_name: ',ffile.split('/')[-1])
            print('min_pehe:%.3f,id:%d\tmax_pehe:%.3f,id:%d\tgap:%.3f'%(min_pehe,min_pehe_idx,max_pehe,max_pehe_idx,max_pehe-min_pehe))
            print('st_idx:%d, end_idx:%d'%(st_idx,end_idx))
    print("\nFinished.\n")
    
if __name__=='__main__':
    if len(sys.argv) < 2:
        print('Usage: python3.6 res_search.py <output_dir>')
    else:
        output_dir = sys.argv[1]
    main(output_dir)