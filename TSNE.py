import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from dataset import GenerateIterator, GenerateIterator_eval
from myargs import args
from sklearn.manifold import TSNE
import pandas as pd
import altair as alt


checkpoint = torch.load('./model_checkpoint.pt')

sharedencoder_net.load_state_dict(checkpoint.sharedencoder_net)


gaussian_net.load_state_dict(checkpoint['gaussian_net'])
classifier_net.load_state_dict(checkpoint['classifier_net'])
fclayerencode_net.load_state_dict(checkpoint['fclayerencode_net'])
flow_net.load_state_dict(checkpoint['flow_net'])
targetdecoder_net.load_state_dict(checkpoint['targetdecoder_net'])






sharedencoder_net.load_state_dict(checkpoint['sharedencoder_net'])
gaussian_net.load_state_dict(checkpoint['gaussian_net'])
classifier_net.load_state_dict(checkpoint['classifier_net'])
fclayerencode_net.load_state_dict(checkpoint['fclayerencode_net'])
flow_net.load_state_dict(checkpoint['flow_net'])
targetdecoder_net.load_state_dict(checkpoint['targetdecoder_net'])

optimizer_sharedencoder_net.load_state_dict(checkpoint['optimizer_enc'])
optimizer_gaussian_net.load_state_dict(checkpoint['optimizer_gau'])
optimizer_classifier_net.load_state_dict(checkpoint['optimizer_cls'])
optimizer_fclayerencode_net.load_state_dict(checkpoint['optimizer_fc'])
optimizer_flow_net.load_state_dict(checkpoint['optimizer_flow'])
optimizer_targetdecoder_net.load_state_dict(checkpoint['optimizer_trgdec'])

sharedencoder_net.eval()
gaussian_net.eval()
classifier_net.eval()
fclayerencode_net.eval()
flow_net.eval()
targetdecoder_net.eval()
# sharedencoder_net.load_state_dict(torch.load('sharedencoder_net.pt'))


with torch.no_grad():
    latent_tar, gts_val = [], []
    for images_target, labels_target in iterator_val:
        images_target, labels_target = images_target.cuda(), labels_target.cuda()

        feats_tar = sharedencoder_net(images_target)
        latent_tar = fclayerencode_net(feats_tar)

        latent_tar = latent_tar.cpu()
        latent_tar.extend(latent_tar)
        gts_val.extend(labels_target)


latent_embedded = TSNE(n_components=2).fit_transform(latent_tar.detach())
alt.renderers.enable('notebook')
df = pd.DataFrame({'x': z_embedded[:,0], 'y': latent_embedded[:,1], 'Class':gts_val+1})
alt.Chart(df).mark_point().encode(alt.X('x'), 
                                  alt.Y('y'), 
                                  alt.Color('Class:N'), 
                                  alt.Tooltip(['Class:N', 'x:Q', 'y:Q'])).interactive()




plt.hist(all_loss[0][-1].numpy(), bins=20, range=(0., 1.), edgecolor='black', color='g')
plt.xlabel('loss');
plt.ylabel('number of data')
# plt.grid()
# plt.savefig('%s/histogram_epoch%03d.jpg' % (path_exp,epoch))
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = PIL.Image.open(buf)
image = transforms.ToTensor()(image)
writer_tensorboard.add_image('Histogram/loss_all', image, epoch)
plt.clf()