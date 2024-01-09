# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid

def log(logfile,str):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print(str)

def save_config(fname):
    """ Save configuration """
    flagdict =  {}
    for k in FLAGS:
        flagdict[k] = FLAGS[k].value
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        if FLAGS.sparse>0:
            data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
            x = load_sparse(fname+'.x')
        else:
            data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
            x = data_in[:,5:]

        data['x'] = x
        data['t'] = data_in[:,0:1]
        data['yf'] = data_in[:,1:2]
        data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return S

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_lin_reweight(X,t,p,sample_weight):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    SWt = tf.gather(sample_weight,it)
    SWc = tf.gather(sample_weight,ic)

    mean_control = tf.reduce_mean(SWc*Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(SWt*Xt,reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],axis=0)
    Mt = tf.concat([M,row],axis=0)
    Mt = tf.concat([Mt,col],axis=1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],axis=0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))],axis=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w

def random_fourier_features(x, w=None, b=None, num_f=1, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    r = int(x.shape[1])
    x = tf.expand_dims(x, axis=-1)
    c = int(x.shape[2])
    if sigma is None or sigma == 0:
        sigma = 1
        w = 1 / sigma * (tf.random_normal(shape=(num_f, c))) # 让每一维原始特征都映射出num_f维RFF特征
        b = 2 * np.pi * tf.random_uniform(shape=(1, r, num_f), maxval=1)

    Z = tf.sqrt(tf.constant(2.0 / num_f))

    mid = tf.einsum('ijk,kl->ijl', x, tf.transpose(w)) # tf.matmul(x, tf.transpose(w))
    mid = tf.add(mid, b)
    mid -= tf.reduce_min(mid, axis=1, keepdims=True)
    mid /= tf.reduce_max(mid, axis=1, keepdims=True)
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (tf.cos(mid) + tf.sin(mid))
    else:
        Z = Z * tf.concat((tf.cos(mid), tf.sin(mid)), axis=-1)
    
    return Z

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = tf.matmul(tf.transpose(x), x) / n # return (r,r), E[X*X]
        e = tf.reshape(tf.reduce_mean(x, axis=0), [-1, 1]) # return (r,1), E[X]
        res = cov - tf.matmul(e, tf.transpose(e)) # return (r,r), E[X*X]-E[X]*E[X]
    else:
        w = tf.reshape(w, [-1, 1])
        cov = tf.matmul(tf.transpose(w * x), x)
        e = tf.reshape(tf.reduce_sum(w*x, axis=0), [-1, 1])
        res = cov - tf.matmul(e, tf.transpose(e))

    return res

def dependence_loss(cfeatures, cweights, name=''):
    all_weights = tf.nn.softmax(cweights, axis=0, name='all_weights_{}'.format(name))
    rff_features = random_fourier_features(cfeatures)
    
    loss = tf.Variable(tf.zeros(shape=(), dtype=tf.float32), name='loss_hsic_{}'.format(name))
    for i in range(rff_features.shape[-1]):
        rff_feature = rff_features[:,:,i]
        hsic_cov = cov(rff_feature, all_weights)
        cov_matrix = hsic_cov * hsic_cov
        loss += (tf.reduce_sum(cov_matrix) - tf.linalg.trace(cov_matrix))
    return loss