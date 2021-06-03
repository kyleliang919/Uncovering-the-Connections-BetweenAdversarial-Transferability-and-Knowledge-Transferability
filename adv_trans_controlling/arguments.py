# MODEL OPTS
def model_args(parser):
    group = parser.add_argument_group('Model', 'Arguments control Model')
    group.add_argument('--arch', default='ResNet', type=str, choices=['ResNet'], 
                       help='model architecture')
    group.add_argument('--depth', default=20, type=int, 
                       help='depth of the model')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the file that contains model checkpoints')
    group.add_argument('--gpu', default='0', type=str, 
                       help='gpu id')
    group.add_argument('--seed', default=0, type=int,
                       help='random seed for torch')


# DATALOADING OPTS
def data_args(parser):
    group = parser.add_argument_group('Data', 'Arguments control Data and loading for training')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128,
                       help='batch size of the train loader')


# BASE TRAINING ARGS
def base_train_args(parser):
    group = parser.add_argument_group('Base Training', 'Base arguments to configure training')
    group.add_argument('--epochs', default=50, type=int,
                       help='number of training epochs')
    group.add_argument('--lr', default=0.1, type=float, 
                       help='learning rate')
    group.add_argument('--sch-intervals', nargs='*', default=[100,150], type=int,
                       help='learning scheduler milestones')
    group.add_argument('--lr-gamma', default=0.1, type=float, 
                       help='learning rate decay ratio')


def transfer_train_args(parser):
    group = parser.add_argument_group('Transfer Training', 'Arguments to configure training')


    group.add_argument('--transfer-coeff', default=1., type=float,
                       help='the coefficient to balance diversity training and standard training')


