import jax.numpy as np
from jax import jit, vmap
import jax 
from jax.lax import scan, cond, fori_loop, dynamic_slice
from jax.util import partial, safe_zip
import pickle 
from time import time 
import numpy as onp
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  if isinstance(batch[0], onp.ndarray):
    return onp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return onp.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return onp.ravel(onp.array(pic, dtype=np.float32))

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
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i = vals
    dists = (WX[:,idx] - WX[curr_i, idx]) / Wdist[curr_i]
    dists = np.where(masks[y,1], dists, np.inf)
    j = np.argmin(dists)
    return cond(dists[j] < best_lb,
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y),
            lambda _: (curr_i, j, dists[j], lbs, idx, WX, Wdist, y),
            None 
    )

def _forifun2(curr_i, carry):
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y = carry


    return cond(lbs[curr_i] < best_lb, 
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist, y),
            _else_branch,
            (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i)
        )    

def _forifun(idx, val):
    lbs_idx, ret, retidx, mindistj, minWdist, WX, Wdist, start, ppc,y = val 
    lbs2 = (mindistj[idx] - WX[:, idx])/minWdist

    best_i, best_j, best_lb, lbs, idx, _, _, _ = fori_loop( start, start+ppc, _forifun2,  (-1, -1, -1000000.0, lbs2, idx, WX, Wdist,y))
    ret = ret.at[retidx, :6].set((best_i, best_j, WX[best_i, idx], WX[best_j, idx], Wdist[best_i, best_j], best_lb))
    return lbs_idx, ret, retidx+1, mindistj, minWdist, WX, Wdist, start, ppc,y

def lossfn(X,Y,W,pnorm,masks,ppc,num_classes):
    WW = (W*W).sum(-1)
    XX = (X*X).sum(-1)
    _WX = WW.reshape(-1,1) - 2*W @ X.T + XX.reshape(1,-1)
    Wdist = 2*pairwise_distances(W, W, pnorm)
    minWdist = np.min(Wdist, axis=0)
    retidx = 0
    ret = np.zeros((X.shape[0],6 + X.shape[1]))
    for y in range(num_classes):
        xs = X[Y == y]
        if(xs.shape[0] == 0):
            continue
        WX = _WX[:, Y==y]
        start = y*ppc 

        arr1 = WX[masks[y,0]]
        i  = np.argmin(arr1, axis=0)
        mindisti = arr1[i]

        arr2 =  WX[masks[y,1]]
        j = np.argmin(arr2, axis=0)
        mindistj = np.min(arr2, axis=0)
        i = i+start 
        j = np.where(j < start, j, j+ppc)

        ret = ret.at[retidx:retidx+xs.shape[0], 6:].set(xs)

        lbs_idxs = (mindistj - mindisti)/Wdist[i,j]
        _, ret, retidx, *_ = fori_loop(0, xs.shape[0], _forifun, (lbs_idxs, ret, retidx, mindistj, minWdist, WX, Wdist, start, ppc, y))
    return ret 

def load():
    with open('mnist.pkl', 'rb') as f:
        ds = pickle.load(f)

    Xtrn = ds['training_images']/255.0
    Ytrn = ds['training_labels']
    Xtst = ds['test_images']/255.0
    Ytst = ds['test_labels']

    return (Xtrn, Ytrn), (Xtst, Ytst)

def train(trngen, tst, W, epochs, pnorm, masks, ppc, num_classes, train_eps, test_eps):
    for e in range(epochs):
        acc = 0
        start = time()
        for X, Y in trngen:
            

            loss = lossfn(X,Y,W,pnorm,masks,ppc,num_classes)
            I, J, dIX, dJX, dIJ, lbs =  (loss.T)[:6]
            I = I.astype(np.int32); J =J.astype(np.int32)
            dIJ = dIJ[:, None]; lbs = lbs[:,None]
            X = loss[:,6:]

            acc += sum(lbs > -test_eps)


            gradI = -(W[I]-X)/dIJ - (lbs * W[I] - W[J])/dIJ**2
            gradJ =  (W[J]-X)/dIJ + (lbs * W[I] - W[J])/dIJ**2

            gradI = np.where(lbs.T > -train_eps, gradI.T, 0).T 
            gradJ = np.where(lbs.T > -train_eps, gradJ.T, 0).T 
            grad = np.zeros_like(W)
            grad = grad.at[I].add(gradI)
            grad = grad.at[J].add(gradJ)
            W = W + grad
        X, Y = tst 
        print(lbs)
        _, _, _, _, _,lbs, *_ = lossfn(X,Y,W,pnorm,masks,ppc,num_classes)
        print(f'epoch: {e}, trn acc: {acc/60000}, tst_acc {np.sum(lbs > -test_eps)/10000}, epoch took {time()-start} sec')


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
            
        rng = np.arange(W.shape[0])
        mask1 = (rng >= start) & (rng < end)
        mask2 = ~mask1
        masks.append((mask1, mask2))
    return np.array(masks)    

if(__name__ == '__main__'):
    import numpy as onp
    ppc, num_classes = 100,10

    trn, tst = load()
    W = get_w(*trn, ppc, num_classes)
    masks = get_masks(ppc, num_classes)

    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=5000, num_workers=0)
    
    train(training_generator, tst, W, epochs = 100, pnorm = 2, masks=masks, ppc=ppc, num_classes=num_classes,
            train_eps = 2.5, test_eps = 1.58)

    ppc, num_classes, d, b = 100,9,3*32**2, 2**12

    W = np.array(onp.random.rand(ppc*num_classes, d))
    Xs = np.array(onp.random.rand(b, d))
    Ys = np.array(onp.random.randint(0, num_classes, (b,)))



    for ii in 2**np.arange(5,13):
        for _ in range(3):
            start = time()

            I, J, dI, dJ, dIJ, lbs = lossfn_new(Xs[:ii],Ys[:ii],W,2,masks,ppc,num_classes).T
            grad = np.zeros_like(W)

            print(time()-start)
        print()

