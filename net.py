# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from utils import *

class cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, I, nn):
        self.n = nn[0]
        self.n0 = nn[1]
        self.n1 = nn[2]
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, I)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, I):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        r_lambda1 = FLAGS.p_lambda1

        self.x = x
        self.t = t
        self.y_ = y_
        self.I = I
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        # weights_in = []; biases_in = []
        i0 = tf.to_int32(tf.where(t < 1)[:,0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        self.i0 = i0
        self.i1 = i1

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            self.bn_biases = []
            self.bn_scales = []

        ''' Construct input/representation layers '''
        with tf.name_scope("representation"):
            h_rep, h_rep_norm, weights_in, biases_in = self._build_latent_graph(dim_input, dim_in, dim_out, FLAGS)
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

        ''' Construct ouput layers '''
        with tf.name_scope("output"):
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)
        self.weights_out = weights_out
        self.weights_pred = weights_pred

        ''' Compute sample reweighting '''
        with tf.name_scope("weight"):
            sample_weight = tf.Variable(tf.ones([self.n, 1]), name='sample_weight')
            self.generator_weight = sample_weight
            sample_weight = tf.abs(sample_weight)
            sample_weight = tf.gather(sample_weight, self.I)

        self.sample_weight = sample_weight
        sample_weight_0 = tf.gather(sample_weight, i0)
        sample_weight_1 = tf.gather(sample_weight, i1)

        ''' Construct weighted factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(tf.abs(y_-y))
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)
            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma)
            imb_error = r_alpha*imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            # imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_dist = mmd2_lin_reweight(h_rep_norm,t,p_ipm,sample_weight)
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
            imb_error = r_alpha * imb_dist


        ''' Total error '''
        tot_error = risk
        tot1_error = risk

        if FLAGS.p_lambda>0: # 网络参数的loss
            tot_error = tot_error + r_lambda*self.wd_loss
            tot1_error = tot1_error + r_lambda*self.wd_loss
        
        if FLAGS.p_lambda1>0: # 样本权重的loss
            tot_error = tot_error + r_lambda1 * tf.square(tf.reduce_sum(sample_weight_0)/self.n0 - 1.0)
            tot_error = tot_error + r_lambda1 * tf.square(tf.reduce_sum(sample_weight_1)/self.n1 - 1.0)

        if FLAGS.p_alpha>0: # IPM loss
            tot_error = tot_error + imb_error
        
        if FLAGS.p_gamma1>0: # 预测网络最后一层的HSIC loss
            tot_error = tot_error + FLAGS.p_gamma1 * dependence_loss(self.h0_pred, sample_weight_0, name = 'pred0_last')
            tot_error = tot_error + FLAGS.p_gamma1 * dependence_loss(self.h1_pred, sample_weight_1, name = 'pred1_last')
        
        if FLAGS.p_gamma2>0: # 表征网络最后一层的HSIC loss
            tot_error = tot_error + FLAGS.p_gamma2 * dependence_loss(h_rep_norm, sample_weight, name = 'rep_last')
        
        if FLAGS.p_gamma3>0: # 其余隐藏层的HSIC loss
            for i in range(len(self.hidden_reps)):
                tot_error = tot_error + FLAGS.p_gamma3 * dependence_loss(self.hidden_reps[i], sample_weight, name = 'hidden_rep%d'%(i+1))
            
            for i in range(len(self.hidden0_preds)):
                tot_error = tot_error + FLAGS.p_gamma3 * dependence_loss(self.hidden0_preds[i], sample_weight_0, name = 'hidden0_pred%d'%(i+1))
                tot_error = tot_error + FLAGS.p_gamma3 * dependence_loss(self.hidden1_preds[i], sample_weight_1, name = 'hidden1_pred%d'%(i+1))


        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.tot1_loss = tot1_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
    
    def _build_latent_graph(self, dim_input, dim_in, dim_out, FLAGS):
        weights_in = [];
        biases_in = []

        h_in = [self.x]
        for i in range(0, FLAGS.n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0 / dim_input * tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.multiply(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        self.bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        self.bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, self.bn_biases[-1], self.bn_scales[-1],1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], self.do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep
        
        self.hidden_reps = h_in[1:FLAGS.n_in] # 表征网络除最后一层外的隐藏层

        return h_rep, h_rep_norm, weights_in, biases_in

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        # return y, h_pred, weights_out, weights_pred
        return y, h_pred, weights_out, weights_pred, biases_out, bias_pred, h_out[1:FLAGS.n_out]

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = self.i0
            i1 = self.i1

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            # y0, h0_pred, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            # y1, h1_pred, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)
            y0, h0_pred, weights_out0, weights_pred0, biases_out0, bias_pred0, h0_out = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, h1_pred, weights_out1, weights_pred1, biases_out1, bias_pred1, h1_out = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            h_pred = tf.dynamic_stitch([i0, i1], [h0_pred, h1_pred])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
            biases_out = biases_out0 + biases_out1
            bias_pred = bias_pred0 + bias_pred1
        else:
            h_input = tf.concat(1,[rep, t])
            y, h_pred, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        # self.h_pred = h_pred # 预测网络最后一层
        self.h0_pred = h0_pred
        self.h1_pred = h1_pred
        self.hidden0_preds = h0_out # 预测网络除最后一层外的隐藏层
        self.hidden1_preds = h1_out # 预测网络除最后一层外的隐藏层
        
        return y, weights_out, weights_pred, biases_out, bias_pred