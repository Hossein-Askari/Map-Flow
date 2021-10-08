import argparse

parser = argparse.ArgumentParser(description="Demo for Training MapFlow model")

######################## Model parameters ########################

parser.add_argument('--classes', default=10, type=int, help='# of classes')
parser.add_argument('--dset_dir', dest='dset_dir', default="./data", help='Where you store the dataset.')
parser.add_argument('--epochs', dest='epochs', default=401, type=int, help='Number of epochs to train on.')
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, help='Number of examples per batch.')
parser.add_argument('--device', dest = 'device',default=0, type=int, help='Index of device')
parser.add_argument('--savedir', dest='savedir', default="./saved_models/mnisttosvhn/mapflow/", help="Where to save the trained model.")

#Auto Encoder settings:
parser.add_argument('--num_latent',  default=128, type=int, help="dimension of latent code z")
# parser.add_argument('--n_features',  default=64, type=int, help="conv input")
parser.add_argument('--beta', dest='beta', default=400, help="weight of reconstruction loss")
parser.add_argument('--image_size',  default=28, type=int, help="size of training image")

#Flow settings:
parser.add_argument('--fc_dim',  default=512, type=int, help="dimension of FC layer in the flow")
parser.add_argument('--num_block',  default=12, type=int, help="number of affine coupling layers in the flow")

#optimization settings
parser.add_argument('--lr', default=3e-4, dest='lr', type=float, help="Learning rate for ADAM optimizer. [0.001]")
parser.add_argument('--beta1', default=0.8, dest='beta1', type=float, help="beta1 for adam optimizer")
parser.add_argument('--beta2', default=0.9, dest='beta2', type=float, help="beta2 for adam optimizer")
parser.add_argument('--decay', default=100, dest='decay', type=float, help="number of epochs to decay the lr by half")
parser.add_argument('--num_workers', dest="num_workers", default=8, type=int, help="Number of workers when load in dataset.")

#datasets settings
parser.add_argument('--source', default='svhn', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--target', default='mnist', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--dataset_name', default='mnisttosvhn', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--channels',  default=3, type=int, help="size of training image")


args = parser.parse_args()



