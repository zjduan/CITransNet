import logging
from time import time
import datetime
from trainer import Trainer

if __name__ == '__main__':
    from base import parser, str2bool
    parser.add_argument('--data_dir', type=str, default='../data_multi/10t10t_5x10/')
    parser.add_argument('--training_set', type=str, default='training_200000')
    parser.add_argument('--test_set', type=str, default='test_5000')
    parser.add_argument('--n_bidder_type', type=int, default=10)
    parser.add_argument('--n_item_type', type=int, default=10)
    parser.add_argument('--d_emb', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)

    parser.add_argument('--n_bidder', type=int, default=5)
    parser.add_argument('--n_item', type=int, default=10)
    parser.add_argument('--r_train', type=int, default=25, help='Number of steps in the inner maximization loop')
    parser.add_argument('--r_test', type=int, default=200, help='Number of steps in the inner maximization loop when testing')
    parser.add_argument('--gamma', type=float, default=1e-3, help='The learning rate for the inner maximization loop')
    parser.add_argument('--n_misreport_init', type=int, default=100)
    parser.add_argument('--n_misreport_init_train', type=int, default=1)

    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_epoch', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--batch_test', type=int, default=30)
    parser.add_argument('--eval_freq', type=int, default=30)
    parser.add_argument('--save_freq', type=int, default=1)

    parser.add_argument('--lamb', type=float, default=5)
    parser.add_argument('--lamb_update_freq', type=int, default=2)
    parser.add_argument('--rho', type=float, default=1)
    parser.add_argument('--delta_rho', type=float, default=5)
    parser.add_argument('--v_min', type=float, default=0)
    parser.add_argument('--v_max', type=float, default=1)
    parser.add_argument('--data_parallel', type=str2bool, default=True)
    parser.add_argument('--continuous_context', type=str2bool, default=False)
    parser.add_argument('--cond_prob', type=str2bool, default=False)
    parser.add_argument('--data_parallel_test', action='store_true')
    parser.add_argument('--test_ckpt_tag', type=str, default=None)

    t0 = time()
    args = parser.parse_args()
    trainer = Trainer(args)
    if args.test:
        trainer.test(args, load=True)
    else:
        trainer.train(args)

    time_used = time() - t0
    logging.info(f'Time Cost={datetime.timedelta(seconds=time_used)}')