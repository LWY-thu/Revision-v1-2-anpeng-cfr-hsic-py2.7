import sys
import os

import pickle # python 3

from logger import Logger as Log
Log.VERBOSE = True

import evaluation as evaluation
from plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None, stop_set = None, stop_criterion=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)

    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+cfg['dataform']
    dataname_test = cfg['data_test'].split('~')
    data_test = [cfg['datadir'] + fi for fi in dataname_test]

    if 'loss' in cfg.keys():
        if cfg['loss'] == 'log':
            y_is_binary = 1
        else:
            y_is_binary = 0
    else:
        y_is_binary = cfg['y_is_binary']

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=y_is_binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))
    
    plot_evaluation_data(eval_results, configs, output_dir, filters, stop_set, stop_criterion, y_is_binary)

    # # Print evaluation results
    # if y_is_binary:
    #     plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters, stop_set, stop_criterion)
    # else:
    #     plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters, stop_set, stop_criterion)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python2 evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        early_stop_set = None
        if len(sys.argv)>3:
            early_stop_set = sys.argv[3]
        
        early_stop_criterion = None
        if len(sys.argv)>4:
            early_stop_criterion = sys.argv[4]
        
        filters = None
        if len(sys.argv)>5:
            filters = eval(sys.argv[5])

        evaluate(config_file, overwrite, filters=filters, stop_set = early_stop_set, stop_criterion = early_stop_criterion)
