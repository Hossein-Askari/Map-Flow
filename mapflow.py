import torch
from torch import nn
from torch import distributions
import numpy as np
from tqdm import tqdm
import argparse
import os
from dataset import create_dataset, create_dataloader
from models import *
from vat import ConditionalEntropyLoss, VAT
from utils import EMA
from myargs import args

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom, GLOWCouplingBlock
from torch.utils.tensorboard import SummaryWriter

def gaussian_loglkhd(h,device):

	return - 0.5*torch.sum(torch.pow(h,2),dim=1) - h.size(1)*0.5*torch.log(torch.tensor(2*np.pi).to(device))

# ------------------------------------------------------------------------------
# device and seeds
# ------------------------------------------------------------------------------

use_gpu = torch.cuda.is_available()
torch.manual_seed(123)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")

# ------------------------------------------------------------------------------
# dataloader
# ------------------------------------------------------------------------------

train_set = create_dataset(args, is_train=True)
iterator_train = create_dataloader(train_set, args, is_train=True)

val_set = create_dataset(args, is_train=False)
iterator_val = create_dataloader(val_set, args, is_train=False)

# ------------------------------------------------------------------------------
# flow models
# ------------------------------------------------------------------------------

def subnet_fc(c_in, c_out):
	return nn.Sequential(nn.Linear(c_in, args.fc_dim), 
						 nn.ReLU(), 
						 nn.BatchNorm1d(args.fc_dim),
						 nn.Linear(args.fc_dim, args.fc_dim), 
						 nn.ReLU(), 
						 nn.BatchNorm1d(args.fc_dim), 
						 nn.Linear(args.fc_dim,  c_out))

nodes = [InputNode(args.num_latent, name='input')]
for k in range(args.num_block):
	nodes.append(Node(nodes[-1], RNVPCouplingBlock, {'subnet_constructor':subnet_fc, 'clamp':2.0}, name=F'coupling_{k}'))
	nodes.append(Node(nodes[-1], PermuteRandom, {'seed':k}, name=F'permute_{k}'))    
nodes.append(OutputNode(nodes[-1], name='output'))
Flow_source = ReversibleGraphNet(nodes, verbose=False)
flows = Flow_source.to(device)


nodes = [InputNode(args.num_latent, name='input')]
for k in range(args.num_block):
	nodes.append(Node(nodes[-1], GLOWCouplingBlock, {'subnet_constructor':subnet_fc, 'clamp':2.0}, name=F'coupling_{k}'))
	nodes.append(Node(nodes[-1], PermuteRandom, {'seed':k}, name=F'permute_{k}'))    
nodes.append(OutputNode(nodes[-1], name='output'))
Flow_target = ReversibleGraphNet(nodes, verbose=False)
flowt = Flow_target.to(device)

# ------------------------------------------------------------------------------
# network initialization
# ------------------------------------------------------------------------------

encoder = Encoder(args).to(device)
galayer = GAlayer(args).to(device)
decoder = Decoder(args).to(device)
fclayer = FClayer(args).to(device)
classifier = Classifier(args).to(device)

# ------------------------------------------------------------------------------
# optimizer 
# ------------------------------------------------------------------------------

trainable_parameterss = [p for p in flows.parameters() if p.requires_grad==True]
optimizer_flows = torch.optim.Adam(trainable_parameterss, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-6, weight_decay=2e-5)
for param in trainable_parameterss:
	param.data = 0.05*torch.randn_like(param)


trainable_parameterst = [p for p in flowt.parameters() if p.requires_grad==True]
optimizer_flowt = torch.optim.Adam(trainable_parameterst, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-6, weight_decay=2e-5)
for param in trainable_parameterst:
	param.data = 0.05*torch.randn_like(param)


optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_gal = torch.optim.Adam(galayer.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_fcl = torch.optim.Adam(fclayer.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# ------------------------------------------------------------------------------
# scheduler 
# ------------------------------------------------------------------------------

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_enc, args.decay, gamma=0.5, last_epoch=-1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_gal, args.decay, gamma=0.5, last_epoch=-1)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_dec, args.decay, gamma=0.5, last_epoch=-1)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer_fcl, args.decay, gamma=0.5, last_epoch=-1)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer_cls, args.decay, gamma=0.5, last_epoch=-1)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer_flows, args.decay, gamma=0.5, last_epoch=-1)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer_flowt, args.decay, gamma=0.5, last_epoch=-1)

# ------------------------------------------------------------------------------
# loss functions
# ------------------------------------------------------------------------------

crent = nn.CrossEntropyLoss(reduction ='sum')
coent = ConditionalEntropyLoss()
recon = nn.MSELoss(reduction = 'sum')

# ------------------------------------------------------------------------------
# loss weight params and tesnorboard-runs paths
# ------------------------------------------------------------------------------

wrs = 1
wls = 1
wen = 1
wcl = 1
wrt = 1
wlt = 1
wco = 1

experiment = '07label'
writer_tensorboard = SummaryWriter('tb_loggers/' + experiment)

# ------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------

if not os.path.exists(args.savedir):
	os.makedirs(args.savedir)  


encoder.train()
galayer.train()
decoder.train()
fclayer.train()
classifier.train()
flowt.train()
flows.train()


for epoch in range(1, args.epochs):
	
	iterator_train.dataset.shuffledata()
	progress_bar = tqdm(iterator_train)

	recon_losses_t = []
	logdt_t = []
	like_losses_t = []
	coent_losses =[]

	recon_losses_s = []
	logdt_s = []
	like_losses_s = []
	ent_losses = []
	classifier_losses =[]   

	predects_src = []
	labels_src = []

	for images_source, labels_source, images_target, labels_target in progress_bar:

		images_source = images_source.to(device)
		labels_source = labels_source.to(device)
		images_target = images_target.to(device)
		labels_target = labels_target.to(device)

		###################################################################
		###################################################################

		optimizer_enc.zero_grad()
		optimizer_gal.zero_grad()
		optimizer_dec.zero_grad()
		optimizer_fcl.zero_grad()
		optimizer_cls.zero_grad()
		optimizer_flows.zero_grad()
		optimizer_flowt.zero_grad()


		feats_trg = encoder(images_target)
		latent_trg = fclayer(feats_trg)
		reconstructed_trg = decoder(latent_trg)

		zhat_trg = flowt(latent_trg.data)
		logd_trg = flowt.log_jacobian(latent_trg.data)

		predect_trg = classifier(latent_trg)

		loss_recon_t = recon(reconstructed_trg, images_target)
		loss_ll_t = gaussian_loglkhd(zhat_trg,args.device) + logd_trg
		loss_ll_t =  -loss_ll_t.mean()
		# loss_coent = crent(predect_trg, labels_target) # check agaian baby
		loss_coent = coent(predect_trg)

		recon_losses_t.append(loss_recon_t.item()/images_target.shape[0])
		logdt_t.append(torch.mean(logd_trg).item())
		like_losses_t.append(loss_ll_t.item())
		coent_losses.append(loss_coent.item()) # check agaian baby

		###################################################################
		###################################################################

		feats_src = encoder(images_source)
		# import pdb; pdb.set_trace()

		mu, logvar, latent_src = galayer(feats_src)
		reconstructed_src = decoder(latent_src)
		zhat_src = flows(latent_src)
		logd_src = flows.log_jacobian(latent_src)


		# for param in flowt.parameters():
		# 	param.requires_grad = False

		latent_src_prim = flowt(zhat_src,rev=True)



		predect_src = classifier(latent_src_prim)

		loss_recon_s = recon(reconstructed_src, images_source)
		loss_ll_s = gaussian_loglkhd(zhat_src, args.device) + logd_src
		loss_ll_s = -loss_ll_s.mean()
		loss_en = -torch.mean(torch.sum(logvar,1)).div(2)
		loss_cl = crent(predect_src, labels_source)


		logdt_s.append(torch.mean(logd_src).item())
		like_losses_s.append(loss_ll_s.item())
		recon_losses_s.append(loss_recon_s.item()/images_source.size(0))
		ent_losses.append(loss_en.item())
		classifier_losses.append(loss_cl.item())


		predect_src = np.argmax(predect_src.cpu().data.numpy(), 1)
		predects_src.extend(predect_src)
		labels_src.extend(labels_source)


		###################################################################
		###################################################################

		total_loss = (wrs * args.beta *loss_recon_s/images_source.size(0) + 
					  wls * loss_ll_s + 
					  wen * loss_en + 
					  wcl * loss_cl + 
					  wrt * loss_recon_t/images_target.size(0) + 
					  wlt * loss_ll_t + 
					  wco * loss_coent
					  )

		total_loss.backward(retain_graph=True)


		optimizer_enc.step()
		optimizer_gal.step()
		optimizer_dec.step()
		optimizer_fcl.step()
		optimizer_cls.step()
		optimizer_flows.step()
		optimizer_flowt.step()

		# for param in flowt.parameters():
		# 	param.requires_grad = True

		progress_bar.set_postfix(recloss_t= np.mean(recon_losses_t), 
			logdt_t = np.mean(logdt_t),
			likeloss_t = np.mean(like_losses_t),
			coent_loss = np.mean(coent_losses), 
			recloss_s= np.mean(recon_losses_s), 
			logdt_s = np.mean(logdt_s),
			likeloss_s = np.mean(like_losses_s), 
			entloss = np.mean(ent_losses), 
			clsloss = np.mean(classifier_losses)
			)

		progress_bar.update(images_target.size(0))

		###################################################################

		###################################################################

	predects_src = np.asarray(predects_src)
	labels_src = np.asarray(labels_src)
	acc_cls_src = (np.mean(predects_src == labels_src)).astype(np.float)


	scheduler1.step()
	scheduler2.step()
	scheduler3.step()
	scheduler4.step()
	scheduler5.step()
	scheduler6.step()
	scheduler7.step()

###################################################################

###################################################################


	print('Train Epoch: {} reconstruction_t loss: {:.4f} loglikelihood_t loss: {:.4f} logdt_t: {:.4f} cond_entropy loss: {:.4f}'.format(
		epoch, np.mean(recon_losses_t), np.mean(like_losses_t), np.mean(logdt_t), np.mean(coent_losses)))

	print('Train Epoch: {} reconstruction_s loss: {:.4f} loglikelihood_s loss: {:.4f} entropy loss: {:.4f} logdt_s: {:.4f} src_pred loss: {:.4f}'.format(
		epoch, np.mean(recon_losses_s), np.mean(like_losses_s), np.mean(ent_losses), np.mean(logdt_s), np.mean(classifier_losses)))


###################################################################

###################################################################

	if epoch % 1 == 0:

		encoder.eval()
		galayer.eval()
		decoder.eval()
		fclayer.eval()
		classifier.eval()
		flowt.eval()
		flows.eval()

		with torch.no_grad():

			preds_eval = []
			gts_eval = []

			for images_source, labels_source, images_target, labels_target in iterator_val:

				images_target = images_target.to(device)
				labels_target = labels_target.to(device)

				feats_trg = encoder(images_target)
				latent_trg = fclayer(feats_trg)
				pred_eval = classifier(latent_trg)

				pred_eval = np.argmax(pred_eval.cpu().data.numpy(), 1)
				preds_eval.extend(pred_eval)
				gts_eval.extend(labels_target)
			
			preds_eval = np.asarray(preds_eval)
			gts_eval = np.asarray(gts_eval)

			score_cls_val = (np.mean(preds_eval == gts_eval)).astype(np.float)

	print('Train Epoch: {} accuracy_srs: {:.4f} validation_trg: {:.4f}'.format(epoch, acc_cls_src, score_cls_val))

	writer_tensorboard.add_scalar('reconstruction_t loss', np.mean(recon_losses_t), epoch)
	writer_tensorboard.add_scalar('loglikelihood_t loss', np.mean(like_losses_t), epoch)
	writer_tensorboard.add_scalar('logdt_t', np.mean(logdt_t), epoch)
	writer_tensorboard.add_scalar('cond_entropy loss', np.mean(coent_losses), epoch)
	writer_tensorboard.add_scalar('trg_val_accuracy', score_cls_val, epoch)

	writer_tensorboard.add_scalar('reconstruction_s loss', np.mean(recon_losses_s), epoch)
	writer_tensorboard.add_scalar('loglikelihood_s loss', np.mean(like_losses_s), epoch)
	writer_tensorboard.add_scalar('entropy loss', np.mean(ent_losses), epoch)
	writer_tensorboard.add_scalar('logdt_s', np.mean(logdt_s), epoch)
	writer_tensorboard.add_scalar('src_pred loss', np.mean(classifier_losses), epoch)
	writer_tensorboard.add_scalar('acc_cls_src', acc_cls_src, epoch)

	if epoch % 150 ==0:
		torch.save(encoder.state_dict(), os.path.join(args.savedir, 'ENCModel_epo{}'.format(epoch)))
		torch.save(galayer.state_dict(), os.path.join(args.savedir, 'GAModel_epo{}'.format(epoch)))
		torch.save(decoder.state_dict(), os.path.join(args.savedir, 'DECModel_epo{}'.format(epoch)))
		torch.save(flows.state_dict(), os.path.join(args.savedir, 'FLOWModel_s_epo{}'.format(epoch)))
		torch.save(fclayer.state_dict(), os.path.join(args.savedir, 'FCModel_epo{}'.format(epoch)))
		torch.save(flowt.state_dict(), os.path.join(args.savedir, 'FLOWModel_t_epo{}'.format(epoch)))
		torch.save(classifier.state_dict(), os.path.join(args.savedir, 'CLModel_epo{}'.format(epoch)))
