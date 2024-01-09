# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

import net as cfr
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 禁用GPU


''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'log', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 0.05, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0001, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_lambda1', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_gamma1', 0, """coef of pred_last_hsic. """)
tf.app.flags.DEFINE_float('p_gamma2', 0, """coef of rep_last_hsic. """)
tf.app.flags.DEFINE_float('p_gamma3', 0, """coef of else_hsic. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.0005, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 256, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 128, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 64, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 1, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('n_experiments', 10, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 5000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
# tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('outdir', './results/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', './Syn_8_8_8_2_10000/syn_conty/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'rp30.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'rn30.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 888, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True

def pehe(ypred1, ypred0, mu1, mu0):
    return np.sqrt(np.mean(np.square((mu1 - mu0) - (ypred1 - ypred0))))


def train(CFR, sess, train_first, train_second, D, I_valid, D_test, logfile, i_exp):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0];n_test = D_test[0]['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid)); I_test = range(n_test)
    n_train, n_valid = len(I_train), len(I_valid)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.I: I_train, CFR.x: D['x'][I_train,:], CFR.t: D['t'][I_train,:], CFR.y_: D['yf'][I_train,:], \
      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.I: I_valid, CFR.x: D['x'][I_valid,:], CFR.t: D['t'][I_valid,:], CFR.y_: D['yf'][I_valid,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
          CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CFR.I: I_train, CFR.x: D['x'][I_train,:], CFR.t: 1-D['t'][I_train,:], CFR.y_: D['ycf'][I_train,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0}
    
    ''' 测试集dict '''
    dict_test = [{CFR.I: I_test, CFR.x: D_test[dt_i]['x'], CFR.t: D_test[dt_i]['t'], CFR.y_: D_test[dt_i]['yf'], \
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated} for dt_i in range(len(D_test))]

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = {di: [] for di in range(len(D_test))}

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        batch_size = FLAGS.batch_size if FLAGS.batch_size else n_train
        I_batch = random.sample(range(0, n_train), batch_size)

        x_batch = D['x'][I_train,:][I_batch,:]
        t_batch = D['t'][I_train,:][I_batch]
        y_batch = D['yf'][I_train,:][I_batch]
        if D['HAVE_TRUTH']:
            yc_batch = D['ycf'][I_train, :][I_batch]

        if __DEBUG__:
            M = sess.run(cfr.pop_dist(CFR.x, CFR.t), feed_dict={CFR.x: x_batch, CFR.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_first, feed_dict={CFR.I: I_batch, CFR.x: x_batch, CFR.t: t_batch, \
                CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated})
            
            sess.run(train_second, feed_dict={CFR.I: I_batch, CFR.x: x_batch, CFR.t: t_batch, \
                CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual)

            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)
            
            ''' 测试集结果 '''
            dt_i = 0
            # test_obj, test_f_error, test_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_test[0])

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                            CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_cf = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                            CFR.t: 1-D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_mu1 = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                            CFR.t: 1-D['t']+D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_mu0 = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                            CFR.t: D['t']-D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            ate_train = np.mean(y_pred_mu1) - np.mean(y_pred_mu0)
            pehe_train = pehe(ypred1=y_pred_f, ypred0=y_pred_cf, mu1=y_pred_mu1, mu0=y_pred_mu0)

            y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                            CFR.t: D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                    CFR.t: 1-D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_mu1_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                            CFR.t: 1-D_test[dt_i]['t']+D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_mu0_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                    CFR.t: D_test[dt_i]['t']-D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            ate_ood = np.mean(y_pred_mu1_test) - np.mean(y_pred_mu0_test)
            pehe_ood = pehe(ypred1=y_pred_f_test, ypred0=y_pred_cf_test, mu1=y_pred_mu1_test, mu0=y_pred_mu0_test)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.4f,\tF: %.4f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f\tate_train: %.4f,\tpehe_train: %.4f\tate_ood: %.4f,\tpehe_ood: %.4f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj,ate_train, pehe_train, ate_ood,pehe_ood)

            # loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f,\tTest: %.3f,\tTestImb: %.2g,\tTestObj: %.2f' \
            #             % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj, test_f_error, test_imb, test_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(CFR.output, feed_dict={CFR.I: I_batch, CFR.x: x_batch, \
                    CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc
                if D['HAVE_TRUTH']:
                    yc_pred = sess.run(CFR.output, feed_dict={CFR.I: I_batch, CFR.x: x_batch, \
                                                              CFR.t: 1 - t_batch, CFR.y_: yc_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                    yc_pred = 1.0 * (yc_pred > 0.5)
                    cacc = 100 * (1 - np.mean(np.abs(yc_batch - yc_pred)))
                    loss_str += ',\tcAcc: %.2f%%' % cacc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True
                exit()

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            if D['HAVE_TRUTH']:
                y_pred_cf = sess.run(CFR.output, feed_dict={CFR.I: I, CFR.x: D['x'], \
                    CFR.t: 1-D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                for dt_i in range(len(D_test)):
                    y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                            CFR.t: D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                    y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.I: I_test, CFR.x: D_test[dt_i]['x'], \
                                            CFR.t: 1-D_test[dt_i]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
                    preds_test[dt_i].append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.I: I, CFR.x: D['x'], \
                    CFR.do_in: 1.0, CFR.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.I: I_test, CFR.x: D_test['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0})
                    reps_test.append(reps_test_i)
    
    # ''' 隐藏层输出 '''
    # test_pred_hidden = []
    # for di in range(len(D_test)):
    #     pred_hidden = sess.run(CFR.h_pred, feed_dict={CFR.x: D_test[di]['x'], CFR.t: D_test[di]['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
    #     test_pred_hidden.append(pred_hidden)

    # rep_hidden, weight_in, pred_hidden, weight_out, weight_pred = sess.run([CFR.h_rep, CFR.weights_in, CFR.h_pred, CFR.weights_out, CFR.weights_pred], \
    #                                                                         feed_dict={CFR.x: D['x'], CFR.t: D['t'], CFR.do_in: 1.0, CFR.do_out: 1.0})
    # net_dict= {'rep_hidden': rep_hidden, 'weight_in': weight_in, 'pred_hidden': pred_hidden, 'weight_out': weight_out, 'weight_pred': weight_pred, 'test_pred_hidden': test_pred_hidden}

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    # net_dir = outdir+'weight'
    # os.mkdir(net_dir)

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataname_test = FLAGS.data_test.split('~')
        dataform_test = [FLAGS.datadir + fi for fi in dataname_test]
        npzfile_test = [outdir+dtf[:-9]+'_result.test' for dtf in dataname_test]
        print('FLAGS.data_test:',FLAGS.data_test,'len(dataform_test):',len(dataform_test),'dataform_test:',dataform_test)

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + ' '.join(datapath_test))
    print(datapath)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = [load_data(test_data) for test_data in datapath_test]
        n_exp_test = D_test[0]['x'].shape[-1]

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    I  = tf.placeholder("int32", shape=[None, ], name='I')   # weight

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out]
    nn = [D['n'], D['n']-np.sum(D['t']), np.sum(D['t'])]
    CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, I, nn)

    ''' Set up optimizer '''
    first_step = tf.Variable(0, trainable=False, name='first_step')
    second_step = tf.Variable(0, trainable=False, name='second_step')
    first_lr = tf.train.exponential_decay(FLAGS.lrate, first_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
    second_lr = tf.train.exponential_decay(FLAGS.lrate, second_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    first_opt = None
    second_opt = None
    if FLAGS.optimizer == 'Adagrad':
        first_opt = tf.train.AdagradOptimizer(first_lr)
        second_opt = tf.train.AdagradOptimizer(second_lr)
    elif FLAGS.optimizer == 'GradientDescent':
        first_opt = tf.train.GradientDescentOptimizer(first_lr)
        second_opt = tf.train.GradientDescentOptimizer(second_lr)
    elif FLAGS.optimizer == 'Adam':
        first_opt = tf.train.AdamOptimizer(first_lr)
        second_opt = tf.train.AdamOptimizer(second_lr)
    else:
        first_opt = tf.train.RMSPropOptimizer(first_lr, FLAGS.decay)
        second_opt = tf.train.RMSPropOptimizer(second_lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    R_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='representation')
    O_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='output')
    W_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='weight')

    train_first = first_opt.minimize(CFR.tot1_loss, global_step=first_step, var_list=R_vars+O_vars)
    train_second = second_opt.minimize(CFR.tot_loss, global_step=second_step,var_list=W_vars)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = {di: [] for di in range(len(D_test))}
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    ''' Handle repetitions '''
    n_experiments = FLAGS.n_experiments
    if FLAGS.repetitions>1:
        if FLAGS.n_experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.n_experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test_list = []
                    for dt_i in range(len(D_test)):
                        D_exp_test = {}
                        D_exp_test['x']  = D_test[dt_i]['x'][:,:,(i_exp-1)%n_exp_test]
                        D_exp_test['t']  = D_test[dt_i]['t'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
                        D_exp_test['yf'] = D_test[dt_i]['yf'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
                        if D_test[dt_i]['HAVE_TRUTH']:
                            D_exp_test['ycf'] = D_test[dt_i]['ycf'][:,(i_exp-1)%n_exp_test].reshape(-1,1)
                        else:
                            D_exp_test['ycf'] = None
                        D_exp_test['HAVE_TRUTH'] = D_test[dt_i]['HAVE_TRUTH']
                        D_exp_test_list.append(D_exp_test)
            else:
                pass
                # datapath = dataform % i_exp
                # D_exp = load_data(datapath)
                # if has_test:
                #     datapath_test = dataform_test % i_exp
                #     D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
                            train(CFR, sess, train_first, train_second, D_exp, I_valid, D_exp_test_list, logfile, i_exp)
        
        # ''' 保存网络参数 '''
        # np.save('%s/exp%d.npy'%(net_dir,i_exp), net_dict)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        for dt_i in range(len(D_test)):
            all_preds_test[dt_i].append(preds_test[dt_i])
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = {di: [] for di in range(len(D_test))}
            for dt_i in range(len(D_test)):
                out_preds_test[dt_i] = np.swapaxes(np.swapaxes(all_preds_test[dt_i],1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            for dt_i in range(len(D_test)):
                np.savez(npzfile_test[dt_i], pred=out_preds_test[dt_i])

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    print(outdir)
    os.makedirs(outdir,exist_ok=True)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()