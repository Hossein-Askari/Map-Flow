# Map-Flow
Pytorch implementation of our work.

## Dependencies
python==3.7
torch==1.0
tqdm==4.31

For invertible layers, we used FrEIA library
pip install git+https://github.com/VLL-HD/FrEIA.git
git clone https://github.com/VLL-HD/FrEIA.git
cd FrEIA

##  python setup.py develop
### Data
Go to the https://github.com/RuiShu/dirt-t/tree/master/data and use download_mnist.py and download_svhn.py to get required .mat files. Place them under data/mnist/ and data/svhn/ folders. Then running python mfl_train.py should work.
For other data sets, go to their official websites.

### Hyperparameters
If you would to modify models' hyperparameters, number of epochs, batch size,  number of hidden units for invertible network, number of coupling blocks, and other, consider looking to myargs file  for configuration examples.
 

### Run code
To start the training process run  mlf_train.py
For example, for run of a single MapFlow on SVHN -> MNIST, run mlf_train.py. As reported MapFlow improves upon state-of-the-art.



