import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--method', type=str, choices=['pt', 'ft', 'pt_and_ft'], help='pretrain or fine_tune or together')
parser.add_argument('--eval_indict', type=str, default='acc', choices=['acc', 'loss', 'feature_extract'], help='acuarcy or loss or feature_extract')
parser.add_argument('--pt_loss', type=str, default="flip", choices=['triplet', 'flip', 'flip_cls',
                    'temporal_consistency', 'net_mixup', 'mutual_loss', 'instance_discriminative',
                     'DPC', 'TSC', 'TemporalDis'],
                    help='flip loss or triplet loss or filp clissification / flip_cls or mutual_loss or dense predictive coding')
parser.add_argument('--nce', type=int, default=0, help='if equip with nce loss')
parser.add_argument('--save_model', type=str, default='checkpoints/')
parser.add_argument('--data', type=str, help='time')
parser.add_argument('--dataset',default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something_something_v1'])
parser.add_argument('--root', type=str, default="")
parser.add_argument('--arch', default='i3d', type=str, choices=['i3d', 'r3d', 'r2p1d', 'c3d'])
parser.add_argument('--logits_channel', type=int, default=1024, help='channel of the last layers')
parser.add_argument('--train_list', default='data/kinetics_rgb_train_list.txt', type=str)
parser.add_argument('--val_list', default='data/kinetics_rgb_val_list.txt', type=str)
parser.add_argument('--cluster_list', default='data/kinetics_rgb_cluster_train_list.txt', type=str)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--logs_path', type=str, default="../experiments/logs/hmdb51_self_supervised")
parser.add_argument('--cluster_train', type=int, default=0)
parser.add_argument('--stride', default=1, type=int, help='stride of temporal image')
parser.add_argument('--weights', default="", type=str, help='checkpoints')
parser.add_argument('--spatial_size', default='224', choices=['112', '224'], help='the network input size')
parser.add_argument('--data_length', default='64', help='input clip length')
parser.add_argument('--clips', default='4', help='global local clips num')
# ========================= Mutual Learning ==========================
parser.add_argument('--mutual_learning', type=int, default=0)
parser.add_argument('--mutual_num', type=int, default=2)
# ========================= Learing Stragety =========================
parser.add_argument('--dropout', '--do', default=0.64, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--mixup', type=int, help ='if use mixup do data augmentation', default=0)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-7)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 20, 25, 30, 35, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--optim', default='sgd', choices=['sgd', 'adam'])
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
parser.add_argument('--workers', type=int, default=4)

# =====================Runtime Config ==========================
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoints (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# ============================Evaluation=============================
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--clip_size', type=int, default=16)

args = parser.parse_args()
