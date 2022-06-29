from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=96, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=1, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=1, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=1, type='int',
                  help='number of data loading workers (default: 16)')

# For data
parser.add_option('--dn', '--data_name', dest='data_name', default='Glum',
                  help='mnist, fashion_mnist, t_mnist, c_mnist (default: mnist)')
parser.add_option('--dl', '--data_len', dest='data_len', default=None, type='int',
                  help='Number of data samples (default: None which automatically takes all samples)')
parser.add_option('--ih', '--img_h', dest='img_h', default=224, type='int',
                  help='input image height (default: 256)')
parser.add_option('--iw', '--img_w', dest='img_w', default=224, type='int',
                  help='input image width (default: 256)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 3)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=1, type='int',
                  help='number of classes (default: 10)')
parser.add_option('--loos', '--loos', dest='loos', default=9, type='int',
                  help='index of the set to leave out for testing (default: 9)')

# For model
parser.add_option('--m', '--model', dest='model', default='densenet',
                  help='vgg, inception, resnet, densenet (default: resnet)')
parser.add_option('--lr', '--lr', dest='lr', default=0.0005, type='float',
                  help='learning rate(default: 0.001)')
parser.add_option('--penalty_weight', type=float, default=5000.0,
                  help='penalty_weight')
parser.add_option('--penalty_anneal_iters', type=int, default=100,
                    help='penalty_anneal_iters')

# For directories
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--gc', '--gradcam', dest='gradcam', default=True,
                  help='whether to run gradcam when running inference or not (default: True)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='./'
                          './save/20220519_132817/models/264',  #263
                  help='path to load a .ckpt model')


options, _ = parser.parse_args()

