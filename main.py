import jax.numpy as np
from jax import jit
from jax.lax import cond, fori_loop
from jax.util import partial
import pickle 
from time import time 
import numpy as onp
from matplotlib import pyplot as plt 
import torch 
from torchvision.transforms import functional as tvF
import scipy.io


class NumpyLoader():
    def __init__(self, X, Y, batch_size=1, shift = None):
        if(shift):
            if(X.shape[-1] != 28**2):
                raise NotImplementedError

            X = torch.tensor(X)
            _X = onp.zeros((X.shape[0], (2*shift+1)**2, X.shape[1]))
            idx =  0
            X = X.reshape(-1,28,28)

            for Tx in range(-shift,1+shift):
                for Ty in range(-shift,1+shift):
                    _X[:, idx] = onp.array(tvF.crop(X, Tx, Ty, 28,28).reshape(X.shape[0], -1))
                    idx += 1
            X = _X
        else:
            X = X[:,None,:]

        self.X = X
        self.Y = Y
        self.batch_size = batch_size 
        self.curr_idx = 0
        self.inds = onp.arange(X.shape[0])
        onp.random.shuffle(self.inds)

    def __iter__(self):
        return self 

    def __next__(self):
        if(self.curr_idx*self.batch_size >= self.X.shape[0]):
            self.curr_idx = 0
            onp.random.shuffle(self.inds)
            raise StopIteration
        else:
            augment = onp.random.randint(self.X.shape[1], size=self.batch_size)
            inds = self.inds[self.curr_idx * self.batch_size : (1+self.curr_idx) * self.batch_size]
            self.curr_idx += 1
            return np.array(self.X[inds, augment]), np.array(self.Y[inds])
    def size(self):
        return self.X.shape[0]

@jit
def _l1(a,b):
    return np.abs(a-b).sum()

vec_l1 = jit(np.vectorize(_l1, signature='(k),(k)->()'))

@jit
def _linf(a,b):
    return np.abs(a-b).max()

vec_linf = jit(np.vectorize(_linf, signature='(k),(k)->()'))

@partial(jit, static_argnums=(2,))
def pairwise_distances(Xs, Ys, pnorm):
    """
        Computes pnorm {1,2,np.inf} distances of vectors form Xs and Ys.

        shapes in:
           Xs: m x d
           Ys: n x d
        shape out:
            m x n 
    """
    if(pnorm == 1):
        Xs = Xs[:,None,:]
        Ys = Ys[None,:,:]
        return vec_l1(Xs,Ys)
    elif(pnorm == np.inf):
        Xs = Xs[:,None,:]
        Ys = Ys[None,:,:]
        return vec_linf(Xs,Ys)
    elif(pnorm == 2):
        XX = (Xs**2).sum(-1).reshape(-1,1) 
        YY = (Ys**2).sum(-1).reshape(1,-1)
        return np.abs(XX + YY - 2* Xs @ Ys.T )**0.5

def _else_branch(vals):
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i, masks = vals
    dists = (WX[:,idx] - WX[curr_i, idx]) / Wdist[curr_i]
    dists = np.where(masks[y,1], dists, np.inf)
    j = np.argmin(dists)
    return cond(dists[j] < best_lb,
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y,masks),
            lambda _: (curr_i, j, dists[j], lbs, idx, WX, Wdist, y, masks),
            None 
    )

def _forifun2(curr_i, carry):
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, masks = carry
    
    return cond(lbs[curr_i] < best_lb, 
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist, y, masks),
            _else_branch,
            (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i, masks)
        )    

def _forifun(idx, val):
    lbs_idx, ret, retidx, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc,y,masks = val 
    lbs2  = (mindistj[idx] - WX[:, idx])/minWdist
    lbs2_ = (mindistj[idx] - WX[:, idx])/maxWdist 
    lbs2 = np.maximum(lbs2, lbs2_)

    best_i, best_j, best_lb, lbs, idx, *_ = fori_loop( start, start+ppc, _forifun2,  (-1, -1, -np.inf, lbs2, idx, WX, Wdist,y, masks))
    ret = ret.at[retidx, :6].set((best_i, best_j, WX[best_i, idx], WX[best_j, idx], Wdist[best_i, best_j], best_lb))
    return lbs_idx, ret, retidx+1, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc,y, masks

def lb_fn(X,Y,W,pnorm,masks,ppc,num_classes):
    WW = (W*W).sum(-1)
    XX = (X*X).sum(-1)
    _WX = WW.reshape(-1,1) - 2*W @ X.T + XX.reshape(1,-1)
    Wdist = 2*pairwise_distances(W, W, pnorm)
    minWdist = np.min(Wdist, axis=0)
    maxWdist = np.max(Wdist, axis=0)
    retidx = 0
    ret = np.zeros((X.shape[0],6 + X.shape[1]))
    for y in range(num_classes):
        xs = X[Y == y]
        if(xs.shape[0] == 0):
            continue
        WX = _WX[:, Y==y]
        rng = np.arange(xs.shape[0])
        start = y*ppc 
        arr1 = WX[masks[y,0]]
        i  = np.argmin(arr1, axis=0)
        mindisti = arr1[i,rng]

        arr2 =  WX[masks[y,1]]
        j = np.argmin(arr2, axis=0)
        mindistj = arr2[j, rng]
        i = i+start 
        j = np.where(j < start, j, j+ppc)

        ret = ret.at[retidx:retidx+xs.shape[0], 6:].set(xs)

        lbs_idxs = (mindistj - mindisti)/Wdist[i,j]
        _, ret, retidx, *_ = fori_loop(0, xs.shape[0], _forifun, (lbs_idxs, ret, retidx, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc, y, masks))
    return ret 

def load(dataset):
    if(dataset == 'mnist'):
        with open('mnist.pkl', 'rb') as f:
            ds = pickle.load(f)

        Xtrn = ds['training_images']/255.0
        Ytrn = ds['training_labels']
        Xtst = ds['test_images']/255.0
        Ytst = ds['test_labels']

        return (Xtrn, Ytrn), (Xtst, Ytst)

def plot_w(W, ppc, num_classes):
    if(W.shape[-1] == 28*28): # mnist
        fig, ax = plt.subplots(nrows=num_classes, ncols=10, sharex=True, sharey=True,)
        for idx in range(num_classes):    
            for jdx in range(10): 
                ax[idx,jdx].imshow(W[ppc*idx+jdx, :].reshape(28,28), cmap='gray')
        plt.show()
    else:
        raise NotImplementedError
        
def grad_lb(W,X,I,J, dIJ,lbs,train_eps):
    gradI = -(W[I]-X)/dIJ - lbs * (W[I] - W[J])/dIJ**2
    gradJ =  (W[J]-X)/dIJ + lbs * (W[I] - W[J])/dIJ**2

    gradI = np.where(lbs.T < train_eps, gradI.T, 0).T 
    gradJ = np.where(lbs.T < train_eps, gradJ.T, 0).T 

    gradW = np.zeros_like(W)
    gradW = gradW.at[I].add(gradI)
    gradW = gradW.at[J].add(gradJ)
    return gradW


def grad_glvq(W,X,I,J,lbs,train_eps, pnorm):     
    shift = (W[J] - W[I])
    norms = ((shift**2).sum(-1)**0.5)
    AEs = X + train_eps * ( ((W[J] - W[I]).T)/norms).T
    WIX = W[I] - AEs 
    WJX = W[J] - AEs

    dix = np.sum(WIX**2, -1)**0.5 
    djx = np.sum(WJX**2, -1)**0.5 
    dixdjx2 = (dix+djx)**2

    gradI = (-2 * WIX.T * djx / (dix*dixdjx2)).T
    gradJ = ( 2 * WJX.T * dix / (djx*dixdjx2)).T

    #gradI = np.where(lbs.T < train_eps, gradI.T, 0).T 
    #gradJ = np.where(lbs.T < train_eps, gradJ.T, 0).T 

    gradW = np.zeros_like(W)
    gradW = gradW.at[I].add(gradI)
    gradW = gradW.at[J].add(gradJ)
    return gradW



def train(loader, tst, W, epochs, pnorm, masks, ppc, num_classes, train_eps, test_eps, lr):
    for e in range(1,1+epochs):
        if( e%4 == 0):
            lr *= 0.7
        start = time()
        for bidx, (X, Y) in enumerate(loader):

            lowebounds = lb_fn(X,Y,W,pnorm,masks,ppc,num_classes)
            I, J, dIX2, dJX2, dIJ, lbs =  (lowebounds.T)[:6]

            I = I.astype(np.int32); J =J.astype(np.int32)
            dIJ = dIJ[:, None]/2; lbs = lbs[:,None]; X = lowebounds[:,6:]

            #gradW = grad_lb(W, X, I, J, dIJ, lbs, train_eps)
            gradW = grad_glvq(W,X,I,J,lbs,train_eps, pnorm)
            correctly_classified = np.sum(lbs[:,0] > test_eps)


            W = W + gradW/np.sum(gradW*gradW)*lr/10
            #W = W + gradW*lr/X.shape[0]

            print(f'epoch: {e:3d}, batch: {bidx:3d}, batch accuracy: {correctly_classified/X.shape[0]:.4f}, batch_loss: {np.sum(np.maximum(train_eps-lbs, 0)):.2f}')

        X, Y = tst 
        lowebounds = lb_fn(X,Y,W,pnorm,masks,ppc,num_classes)
        lbs =  (lowebounds.T)[5]
        print(f'epoch: {e}, tst_acc {(np.sum(lbs > test_eps)/X.shape[0]):.4f}, epoch took {time()-start:.2f} sec')
    return W

def get_w(X, Y, ppc, num_classes):
    dim = X.shape[-1]
    W = np.zeros((ppc*num_classes, dim))
    for y in range(num_classes):
        x = X[Y == y]
        inds = onp.arange(x.shape[0])
        onp.random.shuffle(inds)
        W = W.at[y*ppc:(y+1)*ppc].set(x[inds[:ppc],:])
    return W

def get_masks( ppc, num_classes):
    masks = []
    for i in range(num_classes):
        start =  i    * ppc 
        end   = (i+1) * ppc
            
        rng = np.arange(ppc*num_classes)
        mask1 = (rng >= start) & (rng < end)
        mask2 = ~mask1
        masks.append((mask1, mask2))
    return np.array(masks)    

if(__name__ == '__main__'):
    ppc, num_classes = 100,10

    trn, tst = load('mnist')
    W = get_w(*trn, ppc, num_classes)

    #db = scipy.io.loadmat('MNIST-TrainTestNonBinary.mat')
    #trn, tst = (db['Xtrain'], db['Ytrain'].reshape(-1)), (db['Xtest'], db['Ytest'].reshape(-1))

    masks = get_masks(ppc, num_classes)


    loader = NumpyLoader(*trn, batch_size = 5000, shift=0)
    
    W = train(loader, tst, W, epochs = 100, pnorm = 2, masks=masks, ppc=ppc, num_classes=num_classes, train_eps = 2.5, test_eps = 1.58, lr=2000)
    plot_w(W, ppc, num_classes)