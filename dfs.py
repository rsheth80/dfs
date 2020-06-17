from learnability import DatasetSVM, Solver
import torch
import yaml, os, argparse

DTYPE = torch.float64
USEMAPPINGFILE = True
USEDATASTATSFILE = True

parser = argparse.ArgumentParser(
                    description='differentiable feature selection',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('dataset', type=str)
parser.add_argument('--dataset_props', metavar='PATH', type=str,
                    default='./datasets.yml',
                    help=('yml file holding values for fn_train, fn_eval, '
                          'ncols, nrows, nrows_test, zero_based, neg_label, '
                          'binary'))
parser.add_argument('--dn_data', type=str, metavar='DIR', default='.',
                    help=('location of train/test files; mappings/datastats '
                          'files stored here'))
parser.add_argument('--device', type=str, help=' ',
                    default='\'cuda\' if available, else \'cpu\'')
parser.add_argument('--order', type=int, default=4, help='{1..12}')
parser.add_argument('--penalty', type=float, default=10, help='(0,infty)')
parser.add_argument('--lr', type=float, default=1e-1, help='Adam learning rate')
parser.add_argument('--epochs', type=float, default=1., help=' ')
parser.add_argument('--workers', type=int, default=4,
                    help='for the pytorch dataloader')
parser.add_argument('--seed', type=int, default=0, help='pytorch random seed')
parser.add_argument('--batch', type=int, default=1000, help='target batchsize')
parser.add_argument('--path_output', type=str, metavar='PATH',
                    default='./dfs.features.NUM_SELECTED_FEATURES.txt',
                    help='output text file with selected features')
parser.add_argument('--mem', type=float, default=1.,
                    help='max device mem (gb) for forward pass')
parser.add_argument('--only-datastats', default=False, action='store_true',
                    help='just compute data statistics, i.e., no selection')

# use a custom pytorch DataSet class to store data (this object is passed to the
# pytorch DataLoader during training)
def load_data(dataset_cfg, dn_data, maxmemgb, seed=0):

    data_train = DatasetSVM(os.path.join(dn_data,dataset_cfg['fn_train']),
                            binary=dataset_cfg['binary'],
                            scale=True,
                            dtype=DTYPE,
                            disk=dataset_cfg['disk'],
                            D=dataset_cfg['ncols'],
                            N=dataset_cfg['nrows'],
                            zero_based=dataset_cfg['zero_based'],
                            set_params=False,
                            MAXMEMGB=maxmemgb,
                            n_to_estimate=dataset_cfg['n_for_stats'],
                            seed=seed)
    print('Data loaded.')

    if dataset_cfg['disk'] and USEMAPPINGFILE:
        path_mappings = os.path.join(dn_data,dataset_cfg['name']+'.mappings')
        if not os.path.exists(path_mappings):
            data_train.create_mappings_file(path_mappings,block_size=1000000)
            data_train.load_mappings_file(path_mappings)
            print(len(data_train.mappings),data_train.mappings[-1],
                  data_train.disk_size)
        else:
            data_train.load_mappings_file(path_mappings)

    if USEDATASTATSFILE:
        fn_datastats = dataset_cfg['name']+'_seed%d.datastats'%seed
        path_datastats = os.path.join(dn_data,fn_datastats)
        if not os.path.exists(path_datastats):
            print('Calculating data stats with n_to_estimate=%d'
                  %dataset_cfg['n_for_stats'])
            data_train.init_data_stats()
            data_train.dump_data_stats(path_datastats)
        else:
            data_train.load_data_stats(path_datastats)
    else:
        print('Calculating data stats with n_to_estimate=%d'
              %dataset_cfg['n_for_stats'])
        data_train.init_data_stats()

    return data_train

def dfs(data_train, order, w_penalty, lr, Nbatch_targetsize, Nepochs,
        num_workers, seed, device):

    Nbatch = min(Nbatch_targetsize, data_train.max_batch_size())

    maxiter = Nepochs*(data_train.N//Nbatch)
    accum_steps = -(-Nbatch_targetsize//Nbatch)

    print('Batchsize=%d, target batchsize=%d.'%(Nbatch,Nbatch_targetsize))

    x0 = torch.zeros((data_train.n_features,1), dtype=DTYPE)
    f_squash = lambda x: torch.sigmoid(2*x)
    f_opt = lambda p: torch.optim.Adam(p, lr)

    S = Solver(data_train, order, Nminibatch=Nbatch, x0=x0, C=w_penalty,
               f_squash=f_squash, f_opt=f_opt, accum_steps=accum_steps,
               num_workers=num_workers)
    S.to(device)

    torch.manual_seed(seed) # not strictly needed for repeatability:
                            # when disk=False, training proceeds sequentially
                            # through the data set
    S.train(maxiter)

    n_feats = len(torch.nonzero(S.x>=0)[:,0])
    print(torch.nonzero(S.x>=0)[:,0])
    print('%d features selected at penalty value %g'%(n_feats, w_penalty))

    return S

if __name__=='__main__':

    args = parser.parse_args()

    dataset = args.dataset
    if args.device==parser.get_default('device'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using %s.'%device)
    datasets = yaml.safe_load(open(args.dataset_props,'rt'))
    dn_data = args.dn_data
    order = args.order
    w_penalty = args.penalty
    lr = args.lr
    Nbatch_targetsize = args.batch
    Nepochs = args.epochs
    num_workers = args.workers
    seed = args.seed
    path_output = args.path_output
    maxmemgb = args.mem
    only_datastats = args.only_datastats

    dn_output = os.path.abspath(os.path.dirname(path_output))
    assert os.path.exists(dn_output), '%s does not exist.'%dn_output
    if args.path_output==parser.get_default('path_output'):
        path_output = None

    dataset_cfg = datasets[dataset]
    dataset_cfg.update({'name':dataset})
    data_train = load_data(dataset_cfg, dn_data, maxmemgb)

    if not only_datastats:
        solver = dfs(data_train, order, w_penalty, lr, Nbatch_targetsize,
                     Nepochs, num_workers, seed, device)
        path_features = solver.save_features(path_output)
        print('Selected features saved to %s'%path_features)
