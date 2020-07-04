from option import args
import datetime
from trainer import train_and_eval


def main():
    args.date = datetime.datetime.today().strftime('%m-%d-%H%M')
    if args.method == 'pt':
        args.eval_indict = 'loss'
        train_and_eval(args)
    elif args.method == 'ft':
        args.eval_indict = 'acc'
        train_and_eval(args)
    elif args.method == 'pt_and_ft':
        args.eval_indict = 'loss'
        checkpoints_path = train_and_eval(args)
        if args.pt_loss == 'TemporalDis':
            args.train_list =  '../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt'
            args.val_list = '../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt'
            args.dataset = 'hmdb51'
        args.eval_indict = 'acc'
        args.lr = 0.001
        args.stride = 1
        # args.stride = 4
        args.epochs = 45
        # args.data_length = 16
        args.data_length = 64
        # args.batch_size = 8
        # args.batch_size = 16
        args.batch_size = 4
        args.lr_steps = [10, 20, 25, 30, 35, 40]
        args.weights = checkpoints_path
        train_and_eval(args)
    else:
        Exception("wrong method!")


if __name__ == '__main__':
    main()
