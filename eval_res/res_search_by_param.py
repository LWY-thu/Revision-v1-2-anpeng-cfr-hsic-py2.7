import os
import sys
import glob
import numpy as np


def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def main(output_dir,p_alpha,p_gamma1,p_gamma2):
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]
    for e_dir in exp_dirs:
        cfg = load_config('%s/config.txt' % e_dir)

        # print('p_alpha:%f\tp_gamma1:%f\tp_gamma2:%f'%(cfg['p_alpha'],cfg['p_gamma1'],cfg['p_gamma2']))

        if cfg['p_alpha']==p_alpha and cfg['p_gamma1']==p_gamma1 and cfg['p_gamma2']==p_gamma2:
            print('p_alpha:%f\tp_gamma1:%f\tp_gamma2:%f'%(p_alpha,p_gamma1,p_gamma2))
            print(e_dir)
    
    print('Finished.')
    
if __name__=='__main__':
    if len(sys.argv) < 2:
        print('Usage: python3.6 res_search.py <output_dir> <p_alpha (default 0)> <p_gamma1 (optional)> <p_gamma2 (default 0)>')
    else:
        output_dir = sys.argv[1]

        p_alpha = 0.5
        if len(sys.argv)>2:
            p_alpha = float(sys.argv[2])

        p_gamma1 = 0.0
        if len(sys.argv)>3:
            p_gamma1 = float(sys.argv[3])
        
        p_gamma2 = 0.0
        if len(sys.argv)>4:
            p_gamma2 = float(sys.argv[4])
        
        main(output_dir,p_alpha,p_gamma1,p_gamma2)

