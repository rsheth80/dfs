import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import numpy as np
import time
import pickle
import os
from sklearn.datasets import load_svmlight_file
import scipy.special
import scipy.sparse
import io
import array

#
# data access
#

BYTESPERREAL = 8.
BYTESPERGB = 1024.**3
MAXMEMGB = 1.

class IndexSampler(Sampler):
    def __init__(self, ix):
        self.ix = ix

    def __iter__(self):
        return (i for i in self.ix)

    def __len__(self):
        return len(self.ix)

class DatasetSVM(Dataset):

    def __init__(self, path_data, disk, D=None, N=None, zero_based=None,
                 binary=True, scale=True, n_to_estimate=int(10e3),
                 set_params=True, dtype=torch.get_default_dtype(),
                 MAXMEMGB=MAXMEMGB, seed=None):
        """
        assumes
            0. data is in svm light format (sklearn.datasets.load_svmlight_file)
            1. y can fit into memory; for y with 4.2 billion elements, 31.3 GB
            of memory is necessary @ 8 bytes/scalar.

        for data reading:
            if disk==False, then the svm file is loaded into memory (in sparse
                format) and example/label pairs can be accessed non-sequentially
            if (disk==True AND there is no mappings file), then the
                example/label pairs will always be returned sequentially
            if (disk==True AND the mappings file has been loaded), then the
                example/label pairs can be randomly accessed

        for the estimation of the data statistics,
            if n_to_estimate==0, all data is used
            if (n_to_estimate<0 AND the mappings file has been loaded),
                -n_to_estimate example/label pairs are randomly accessed
            if (n_to_estimate<0 AND there is no mappings file) OR
                n_to_estimate>0, example/label pairs are sequentially accessed
                from the beginning
        """

        self.path_data = path_data
        self.disk = disk
        self.binary = binary
        self.MAXMEMGB = MAXMEMGB
        self.scale = scale
        self.set_params = set_params
        self.dtype = dtype
        self.seed = seed # for reproducible spectral norm estimation

        self.disk_size = os.path.getsize(path_data)

        if self.disk:

            assert ((D is not None) and (N is not None) and
                    (zero_based is not None)), \
                    'When disk=True, D, N, and zero_based must be provided.'
            self.D = D
            self.N = N
            self.zero_based = zero_based

            self.f = open(self.path_data)

        else:

            if (D is not None) or (N is not None) or (zero_based is not None):
                print('D, N, and zero_based are ignored when disk=False.')

            self.zero_based = False

            self.X, self.y = load_svmlight_file(self.path_data)
            self.N, self.D = self.X.shape

        self.n_features = self.D + self.zero_based
        self.max_rows = int(self.MAXMEMGB*BYTESPERGB/BYTESPERREAL/self.D)
        if n_to_estimate==0:
            self.n_to_estimate = self.N
        else:
            self.n_to_estimate = np.minimum(abs(n_to_estimate), self.N)\
                                            *n_to_estimate/abs(n_to_estimate)

        self.set_return_raw(True)
        self.set_return_np(False)

        if self.set_params:
            self.init_data_stats()
            self.set_return_raw(False)

        self.DISPLAY_PERIOD = 1000

    def set_return_np(self, B):

        self.return_np = B

    def set_return_raw(self, B):

        self.return_raw = B

    def have_mappings(self):

        return (hasattr(self, 'mappings') and len(self.mappings))

    def estimate_spectral_norm(self, data_loader, initv, n_power_iters=10):

        if self.disk and not self.have_mappings():
            self.f.seek(fpos)

        self.set_return_raw(False)
        self.sv1 = 1.   # for self.apply_transform()

        sv1_history = np.zeros(n_power_iters)
        oldv = initv
        for k in range(n_power_iters):
            newv = np.zeros_like(oldv)
            for n,(xn,_) in enumerate(data_loader):
                xn = xn.detach().numpy().squeeze()
                newv += xn*np.dot(xn,oldv)

                if not (n+1)%self.DISPLAY_PERIOD:
                    print('Spectral norm: Read %d rows' % (n+1))
            cosang = np.dot(oldv,newv)
            newnorm = np.linalg.norm(newv)
            sv1_history[k] = np.sqrt(cosang/self.n_to_estimate)
            print(k,cosang/newnorm,sv1_history[k])
            oldv = newv/newnorm
            if self.disk and not self.have_mappings():
                self.f.seek(fpos)

        if self.disk and not self.have_mappings():
            self.f.seek(fpos)

        return sv1_history[-1]

    def init_data_stats(self):
        """
        1. computes/estimates feature means
        2. if scale=True, computes/estimates feature standard devs
        3. if not binary, computes/estimates target mean/standard dev
        4. estimates largest singular value of data matrix
        """

        bool_raw = self.return_raw
        bool_np = self.return_np
        self.set_return_raw(True)
        self.set_return_np(True)

        print('Finding data statistics...', flush=True)

        if self.seed is not None:
            np.random.seed(self.seed)

        if (self.n_to_estimate<0 and self.have_mappings()) or (not self.disk):
            ix_range = np.random.permutation(self.N)[:self.n_to_estimate]
        else:
            ix_range = range(self.n_to_estimate)

        initv = np.random.randn(self.n_features)

        if self.disk:

            if not self.have_mappings():
                fpos = 0
                self.f.seek(fpos)
                num_workers = 0
            else:
                num_workers = 4 # probably should not be hard-coded

            dl = DataLoader(self,
                            batch_sampler=BatchSampler(IndexSampler(ix_range),
                                                       batch_size=1,
                                                       drop_last=False),
                            num_workers=num_workers)

            self.Xmn = np.zeros((1,self.n_features))
            y = np.zeros(self.n_to_estimate)
            for n,(Xn,yn) in enumerate(dl):
                self.Xmn += Xn.detach().numpy()
                y[n] = yn.detach().numpy()

                if not (n+1)%self.DISPLAY_PERIOD:
                    print('Centering: Read %d rows' % (n+1))

            self.Xmn /= self.n_to_estimate

            if self.scale:
                if not self.have_mappings():
                    self.f.seek(fpos)

                self.Xsd = np.zeros((1,self.n_features))
                for n,(Xn,_) in enumerate(dl):
                    self.Xsd += (Xn.detach().numpy()-self.Xmn)**2

                    if not (n+1)%self.DISPLAY_PERIOD:
                        print('Scaling: Read %d rows' % (n+1))

                self.Xsd /= (self.n_to_estimate-1)
                self.Xsd = np.sqrt(self.Xsd)
                self.Xsd[self.Xsd==0] = 1.

            else:
                self.Xsd = 1.

            self.sv1 = self.estimate_spectral_norm(dl, initv)

        else:

            X = self.X.toarray()[ix_range]
            self.Xmn = X.mean(axis=0)

            if self.scale:
                self.Xsd = X.std(axis=0)
                self.Xsd[self.Xsd==0] = 1.
            else:
                self.Xsd = 1.

            Xc = (X-self.Xmn)/self.Xsd
            self.sv1 = scipy.sparse.linalg.svds(Xc/np.sqrt(self.n_to_estimate),
                                                k=1, which='LM',
                                                return_singular_vectors=False)

            y = self.y[:self.n_to_estimate]

        if not self.binary:
            self.ymn = y.mean()
            self.ysd = y.std()
        else:
            self.ymn = 0.
            self.ysd = 1.

        self.set_return_raw(bool_raw)
        self.set_return_np(bool_np)

    def apply_transform(self, X, y):

        if not self.binary:
            y = (y.reshape((-1,1)) - self.ymn)/self.ysd
        else:
            y = y.reshape((-1,1))

        X = (X - self.Xmn)/self.sv1

        if self.scale:
            X /= self.Xsd

        return X, y

    def max_batch_size(self):

        return int(np.min([self.max_rows, self.N]))

    def __len__(self):

        return self.N

    def getXy(self, ix):

        if self.disk:
            if self.have_mappings():
                X,y = load_svmlight_file(
                        self.path_data,
                        offset=self.mappings[ix]-(not not ix),
                        length=self.mappings[ix+1]-self.mappings[ix]-(not ix),
                        n_features=self.n_features, zero_based=self.zero_based)
            else:
                if self.f.tell()>=self.disk_size:
                    self.f.seek(0)

                X,y = load_svmlight_file(
                        io.BytesIO(self.f.readline().encode('utf-8')),
                        n_features=self.n_features, zero_based=self.zero_based)

            return X.toarray(), y
        else:
            return self.X[ix], self.y[ix]

    def __getitem__(self, ix):

        X, y = self.getXy(ix)

        if not self.return_raw:
            X, y = self.apply_transform(X, y)

        if self.binary:
            y[y==0] = -1

        if self.return_np:
            return np.asarray(X).reshape((-1)), y

        X = torch.tensor(X, dtype=self.dtype, requires_grad=False).reshape((-1))
        y = torch.tensor(y, dtype=self.dtype, requires_grad=False).reshape((-1))

        return X, y

    def dump_data_stats(self, fn):

        pickle.dump([self.Xmn,self.Xsd,self.sv1,self.ymn,self.ysd],
                    open(fn,'wb'))
        print('Dumped data statistics to %s.'%fn)

    def load_data_stats(self, fn):

        self.Xmn,self.Xsd,self.sv1,self.ymn,self.ysd \
                                                    = pickle.load(open(fn,'rb'))
        self.set_return_raw(False)
        print('Loaded data statistics from %s.'%fn)

    def create_mappings_file(self, fn, block_size=1000):

        print('Creating mappings file...')

        fw = open(fn, 'wb')
        b = array.array('q',[])
        fpos = self.f.tell()
        self.f.seek(0)
        rows_read_total = 0
        while True:
            _ = next(iter(self.f.readline,''))
            b.append(self.f.tell())
            if self.f.tell()>=self.disk_size:
                break
            if not (len(b)%block_size):
                rows_read_total += block_size
                print('Read/wrote %d rows' % rows_read_total)
                b.tofile(fw)
                b = array.array('q',[])
        if len(b):
            rows_read_total += len(b)
            print('Read/wrote %d rows' % rows_read_total)
            b.tofile(fw)
        fw.close()
        self.f.seek(fpos)

        print('Dumped mappings to %s.'%fn)

    def load_mappings_file(self, fn):

        self.mappings = array.array('q',[])
        self.mappings.fromfile(open(fn,'rb'), self.N)
        self.mappings.insert(0,0)
        print('Loaded mappings from %s.'%fn)

#
# feature selection
#

# coefficients for sublinear estimator were computed running the sublinear
# paper's authors' code
fn_coeffs = 'coeffs.pickle'
path_coeffs = os.path.join(os.path.dirname(__file__), fn_coeffs)
sle_coeffs = pickle.load(open(path_coeffs, 'rb'))

revcumsum = lambda U: U.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

def triudr(X, r):

    Zr = torch.zeros_like(X, requires_grad=False)
    U = X*r
    Zr[:-1] = X[:-1]*revcumsum(U)[1:]

    return Zr

def triudl(X, l):

    Zl = torch.zeros_like(X, requires_grad=False)
    U = X*l
    Zl[1:] = X[1:]*(U.cumsum(dim=0)[:-1])

    return Zl

class ramp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 1e-2
        grad_input[input > 1] = -1e-2
        return grad_input

class safesqrt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        o = input.sqrt()
        ctx.save_for_backward(input, o)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        input, o = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= 0.5/(o+1e-8)
        return grad_input

def ret_val(z, binary):

    if not binary:
        return 1-z

    else:
         return 0.5*(1 - safesqrt.apply(ramp.apply(z)))

class LearnabilityMB(nn.Module):

    def __init__(self, Nminibatch, D, coeff, binary=False,
                 dtype=torch.get_default_dtype()):
        super(LearnabilityMB, self).__init__()

        a = coeff/scipy.special.binom(Nminibatch, np.arange(coeff.size)+2)
        self.order = a.size
        self.a = torch.tensor(a, dtype=dtype, requires_grad=False)
        self.binary = binary

    def forward(self, s, X, y):

        l = y.clone()
        r = y.clone()
        z = 0

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[0]*p

        if self.order < 2:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[1]*p

        if self.order < 3:
            return ret_val(z, self.binary)

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[2]*p

        if self.order < 4:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[3]*p

        if self.order < 5:
            return ret_val(z, self.binary)

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[4]*p

        if self.order < 6:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[5]*p

        if self.order < 7:
            return ret_val(z, self.binary)

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[6]*p

        if self.order < 8:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[7]*p

        if self.order < 9:
            return ret_val(z, self.binary)

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[8]*p

        if self.order < 10:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[9]*p

        if self.order < 11:
            return ret_val(z, self.binary)

        Z = triudr(X, r)
        r = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[10]*p

        if self.order < 12:
            return ret_val(z, self.binary)

        Z = triudl(X, l)
        l = torch.mm(Z, s)
        p = torch.mm(l.t(), r)
        z += self.a[11]*p

        return ret_val(z, self.binary)

class Solver(nn.Module):

    def __init__(self, ds_train, order, Nminibatch=None, x0=None, C=1,
                 f_squash=torch.sigmoid,
                 f_opt=lambda p: torch.optim.Adam(p, 1e-1),
                 accum_steps=1, max_norm_clip=1., num_workers=0):
        super(Solver, self).__init__()

        self.Ntrain, self.D = ds_train.N, ds_train.n_features

        if Nminibatch is None:
            Nminibatch = self.Ntrain
        else:
            if Nminibatch > self.Ntrain:
                print('Minibatch larger than sample size.'
                      + (' Reducing from %d to %d.'
                         % (Nminibatch, self.Ntrain)))
                Nminibatch = self.Ntrain
        if Nminibatch > ds_train.max_rows:
            print('Minibatch larger than mem-allowed.'
                  + (' Reducing from %d to %d.' % (Nminibatch,
                                                   ds_train.max_rows)))
            Nminibatch = int(np.min([Nminibatch, ds_train.max_rows]))
        self.Nminibatch = Nminibatch
        self.accum_steps = accum_steps

        self.dtype = ds_train.dtype

        if x0 is None:
            x0 = torch.zeros(self.D, 1, dtype=self.dtype)
        self.x = nn.Parameter(x0)
        self.f_squash = f_squash
        self.w = torch.tensor(C/(C+1), dtype=self.dtype, requires_grad=False)
        self.C = C

        self.dl_train = DataLoader(ds_train,
                                   batch_size=self.Nminibatch,
                                   drop_last=True,
                                   num_workers=num_workers,
                                   pin_memory=True)

        self.f_train = LearnabilityMB(self.Nminibatch, self.D,
                                      sle_coeffs[order],
                                      binary=ds_train.binary,
                                      dtype=self.dtype)

        self.opt = f_opt(torch.nn.ParameterList([self.x]))
        self.max_norm = max_norm_clip

        self.it = 0
        self.iters_per_epoch = int(np.ceil(self.Ntrain/self.Nminibatch))

        self.device = 'cpu'

    def to(self, dev):

        super(Solver, self).to(dev)

        self.f_train = self.f_train.to(dev)
        self.w = self.w.to(dev)

        self.device = dev

    def penalty(self, s):

        return torch.sum(s)/self.D

    def train(self, n_iters):

        t = time.time()

        it0 = self.it

        h = torch.zeros([1,1], dtype=self.dtype)
        h = h.to(self.device)
        self.x.grad = torch.zeros_like(self.x)

        flag_stop = False
        dataloader_iterator = iter(self.dl_train)

        while not flag_stop:

            try:
                xsub, ysub = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.dl_train)
                xsub, ysub = next(dataloader_iterator)

            xsub = xsub.to(self.device)
            ysub = ysub.to(self.device)

            try:

                s = self.f_squash(self.x)
                s = s.to(self.device)
                f_train = self.f_train(s, xsub, ysub)
                pen = self.penalty(s)

                grad_outputs = torch.tensor([[1]], dtype=self.dtype,
                                            device=self.device)
                g1, = torch.autograd.grad([f_train], [self.x], grad_outputs,
                                          retain_graph=True)
                grad_outputs = torch.tensor([[1]], dtype=self.dtype,
                                            device=self.device)
                g2, = torch.autograd.grad([pen], [self.x], grad_outputs)
                self.x.grad += ((1-self.w)*g1 + self.w*g2)/self.accum_steps

                h += ((1-self.w)*f_train.detach() + self.w*pen.detach()) \
                                                            /self.accum_steps

                self.it += 1
                if self.it % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                                            torch.nn.ParameterList([self.x]),
                                            max_norm=self.max_norm)
                    self.opt.step()

                    t = time.time() - t

                    if self.it-it0 >= n_iters:
                        flag_stop = True

                    epoch = int(self.it/self.iters_per_epoch)
                    print('[%6d/%3d/%3.3f s] %0.3f' % (self.it, epoch, t, h))

                    if flag_stop:
                        break
                    else:
                        self.opt.zero_grad()
                        h = 0
                        t = time.time()

            except KeyboardInterrupt:
                flag_stop = True
                break

    def save_features(self, path_save=None):

        x = torch.nonzero(self.x>=0)[:,0].detach().cpu().numpy()
        if path_save is None:
            path_save = 'dfs.features.%d.txt' % x.size
        np.savetxt(path_save, x, '%d')

        return path_save
