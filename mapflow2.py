import torch
from torch import nn
from torch import distributions
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import os
from dataset import create_dataset, create_dataloader
# from dataset import GenerateIterator, GenerateIterator_eval
from models import *
from vat import ConditionalEntropyLoss, VAT
from utils import EMA
from myargs import args

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
flow_snet = Flow_source.to(device)

nodes = [InputNode(args.num_latent, name='input')]
for k in range(args.num_block):
	nodes.append(Node(nodes[-1], GLOWCouplingBlock, {'subnet_constructor':subnet_fc, 'clamp':2.0}, name=F'coupling_{k}'))
	nodes.append(Node(nodes[-1], PermuteRandom, {'seed':k}, name=F'permute_{k}'))    
nodes.append(OutputNode(nodes[-1], name='output'))
Flow_target = ReversibleGraphNet(nodes, verbose=False)
flow_tnet = Flow_target.to(device)

# ------------------------------------------------------------------------------
# network initialization
# ------------------------------------------------------------------------------

encoder_net = Encoder_shared(args, track_bn = False).to(device)
fclayer_snet = Fclayer_source(args, track_bn = False).to(device)
decoder_snet = Decoder_source(args, track_bn = False).to(device)
fclayer_tnet = Fclayer_target(args, track_bn = False).to(device)
decoder_tnet = Decoder_target(args, track_bn = False).to(device)
classifier_net = Classifier(args, track_bn = False).to(device)

# flow_tnet = Flow_source(args).to(device)
# flow_snet = Flow_target(args).to(device)
# model_s = Model_source(args).cuda()
# model_t = Model_target(args).cuda()

# ------------------------------------------------------------------------------
# optimizer 
# ------------------------------------------------------------------------------

optimizer_enc = torch.optim.Adam(encoder_net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_srcfcl = torch.optim.Adam(fclayer_snet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_srcdec = torch.optim.Adam(decoder_snet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_trgfcl = torch.optim.Adam(fclayer_tnet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_trgdec = torch.optim.Adam(decoder_tnet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_cls = torch.optim.Adam(classifier_net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainable_parameterss = [p for p in flow_snet.parameters() if p.requires_grad==True]
optimizer_srcflow = torch.optim.Adam(trainable_parameterss, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-6, weight_decay=2e-5)
for param in trainable_parameterss:
	param.data = 0.05*torch.randn_like(param)


trainable_parameterst = [p for p in flow_tnet.parameters() if p.requires_grad==True]
optimizer_trgflow = torch.optim.Adam(trainable_parameterst, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-6, weight_decay=2e-5)
for param in trainable_parameterst:
	param.data = 0.05*torch.randn_like(param)

# ------------------------------------------------------------------------------
# scheduler 
# ------------------------------------------------------------------------------

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_enc, args.decay, gamma=0.5, last_epoch=-1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_srcfcl, args.decay, gamma=0.5, last_epoch=-1)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_srcdec, args.decay, gamma=0.5, last_epoch=-1)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer_trgfcl, args.decay, gamma=0.5, last_epoch=-1)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer_trgdec, args.decay, gamma=0.5, last_epoch=-1)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer_cls, args.decay, gamma=0.5, last_epoch=-1)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer_srcflow, args.decay, gamma=0.5, last_epoch=-1)
scheduler8 = torch.optim.lr_scheduler.StepLR(optimizer_trgflow, args.decay, gamma=0.5, last_epoch=-1)

# ------------------------------------------------------------------------------
# loss functions
# ------------------------------------------------------------------------------

crent = nn.CrossEntropyLoss(reduction ='sum').to(device)
coent = ConditionalEntropyLoss().to(device)
recon = nn.MSELoss(reduction = 'sum').to(device)

# vat_loss_source = VAT(model_s).to(device)
# vat_loss_target = VAT(model_t).to(device)

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

experiment = 'exp100002'
writer_tensorboard = SummaryWriter('tb_loggers/' + experiment)

# ------------------------------------------------------------------------------
# Exponential moving average (simulating teacher model)
# ------------------------------------------------------------------------------

ema = EMA(0.998)
ema.register(encoder_net)
ema.register(fclayer_snet)
ema.register(decoder_snet)
ema.register(fclayer_tnet)
ema.register(decoder_tnet)
ema.register(classifier_net)

# ------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------

if not os.path.exists(args.savedir):
	os.makedirs(args.savedir)  


encoder_net.train()
fclayer_snet.train()
decoder_snet.train()
flow_snet.train()
fclayer_tnet.train()
decoder_tnet.train()
flow_tnet.train()
classifier_net.train()


for epoch in range(1, args.epochs):
	
	iterator_train.dataset.shuffledata()
	progress_bar = tqdm(iterator_train)

	recon_losses_s = []
	like_losses_s = []
	log_DS = []
	entr_L = []
	classifier_losses =[]   

	log_DT = []
	recon_losses_t = []
	like_losses_t = []
	conent_losses =[]
	
	preds_tra_s = []
	gts_tra_s = []

	for images_source, labels_source, images_target, labels_target in progress_bar:

		images_source = images_source.to(device)
		labels_source = labels_source.to(device)
		images_target = images_target.to(device)
		labels_target = labels_target.to(device)

		###################################################################
		###################################################################

		optimizer_enc.zero_grad()
		optimizer_srcfcl.zero_grad()
		optimizer_srcdec.zero_grad()
		optimizer_srcflow.zero_grad()
		optimizer_trgfcl.zero_grad()
		optimizer_trgdec.zero_grad()
		optimizer_trgflow.zero_grad()
		optimizer_cls.zero_grad()


		feats_trg = encoder_net(images_target)
		latent_trg = fclayer_tnet(feats_trg)
		reconstructed_trg = decoder_tnet(latent_trg)
		latent_trg_detached = latent_trg.detach()
		zhat_trg = flow_tnet(latent_trg_detached)
		logd_trg = flow_tnet.log_jacobian(latent_trg_detached)
		predect_trg = classifier_net(latent_trg)

		loss_recon_t = recon(reconstructed_trg, images_target)
		loss_ll_t = gaussian_loglkhd(zhat_trg, args.device) + logd_trg
		loss_ll_t = -loss_ll_t.mean()
		# loss_conent = coent(predect_trg)
		loss_conent = crent(predect_trg, labels_target)


		log_DT.append(torch.mean(logd_trg).item())
		recon_losses_t.append(loss_recon_t.item()/images_target.shape[0])
		like_losses_t.append(loss_ll_t.item())
		conent_losses.append(loss_conent.item())

		###################################################################
		###################################################################

		feats_src = encoder_net(images_source)
		mu, logvar, latent_src = fclayer_snet(feats_src)
		reconstructed_src = decoder_snet(latent_src)
		zhat_src = flow_snet(latent_src)
		logd_src = flow_snet.log_jacobian(latent_src)


		# for param in flow_tnet.parameters():
		# 	param.requires_grad = False
		latent_src_prim = flow_tnet(zhat_src,rev=True)
		pred_tra_s = classifier_net(latent_src_prim)

		# for param in flow_tnet.parameters():
		# 	param.requires_grad = True

		loss_recon_s = recon(reconstructed_src, images_source)
		loss_ll_s = gaussian_loglkhd(zhat_src, args.device) + logd_src
		loss_ll_s = -loss_ll_s.mean()
		loss_en = -torch.mean(torch.sum(logvar,1)).div(2)
		loss_cl = crent(pred_tra_s, labels_source)


		log_DS.append(torch.mean(logd_src).item())
		recon_losses_s.append(loss_recon_s.item()/images_source.shape[0])
		like_losses_s.append(loss_ll_s.item())
		entr_L.append(loss_en.item())
		classifier_losses.append(loss_cl.item())


		pred_tra_s = np.argmax(pred_tra_s.cpu().data.numpy(), 1)
		preds_tra_s.extend(pred_tra_s)
		gts_tra_s.extend(labels_source)


		###################################################################
		###################################################################

		total_loss = (wrs * loss_recon_s/images_source.size(0) + 
					  wls * loss_ll_s + 
					  wen * loss_en + 
					  wcl * loss_cl + 
					  wrt * loss_recon_t/images_target.size(0) + 
					  wlt * loss_ll_t + 
					  wco * loss_conent
					  )

		total_loss.backward(retain_graph=True)

		torch.nn.utils.clip_grad_value_(encoder_net.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(fclayer_snet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(decoder_snet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(flow_snet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(fclayer_tnet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(decoder_tnet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(flow_tnet.parameters(), 1.0)
		torch.nn.utils.clip_grad_value_(classifier_net.parameters(), 1.0)

		ema(encoder_net)
		ema(fclayer_snet)
		ema(decoder_snet)
		ema(fclayer_tnet)
		ema(decoder_tnet)
		ema(classifier_net)


		optimizer_enc.step()
		optimizer_srcfcl.step()
		optimizer_srcdec.step()
		optimizer_srcflow.step()
		optimizer_trgfcl.step()
		optimizer_trgdec.step()
		optimizer_trgflow.step()
		optimizer_cls.step()


		progress_bar.set_postfix(los_rec_s = (loss_recon_s.item()/images_source.shape[0]), 
								 logd_s = torch.mean(logd_src).item(), 
								 likloss_s = loss_ll_s.item(),
								 entrpy = loss_en.item(),
								 clasif = loss_cl.item(),
								 los_rec_t = loss_recon_t.item(),
								 logd_t = torch.mean(logd_trg).item(),
								 likloss_t = loss_ll_t.item(),
								 conent = loss_conent.item())

		progress_bar.update(images_target.size(0))

		###################################################################

		###################################################################

	preds_tra_s = np.asarray(preds_tra_s)
	gts_tra_s = np.asarray(gts_tra_s)
	acc_cls_src = (np.mean(preds_tra_s == gts_tra_s)).astype(np.float)


	scheduler1.step()
	scheduler2.step()
	scheduler3.step()
	scheduler4.step()
	scheduler5.step()
	scheduler6.step()
	scheduler7.step()
	scheduler8.step()

###################################################################

###################################################################


	print('Train Epoch: {} reconstruction_s loss: {:.4f} loglikelihood_s loss: {:.4f}  entropy loss: {:.4f}  log_s det: {:.4f}  source_pred loss: {:.4f}'.format(
		epoch, np.mean(recon_losses_s), np.mean(like_losses_s), np.mean(entr_L), np.mean(log_DS), np.mean(classifier_losses) ))

	print('Train Epoch: {} reconstruction_t loss: {:.4f} loglikelihood_t loss: {:.4f}  log_t det: {:.4f}  cond_entropy loss: {:.4f} Accuracy_source: {:.4f} '.format(
		epoch, np.mean(recon_losses_t), np.mean(like_losses_t), np.mean(log_DT), np.mean(conent_losses), acc_cls_src))



###################################################################

###################################################################

	if epoch % 1 == 0:

		encoder_net.eval()
		fclayer_snet.eval()
		decoder_snet.eval()
		flow_snet.eval()
		fclayer_tnet.eval()
		decoder_tnet.eval()
		flow_tnet.eval()
		classifier_net.eval()

		with torch.no_grad():

			preds_eval = []
			gts_eval = []

			for images_source, labels_source, images_target, labels_target in iterator_val:

				images_target = images_target.to(device)
				labels_target = labels_target.to(device)

				feats_trg = encoder_net(images_target)
				latent_trg = fclayer_tnet(feats_trg)
				pred_eval = classifier_net(latent_trg)

				pred_eval = np.argmax(pred_eval.cpu().data.numpy(), 1)
				preds_eval.extend(pred_eval)
				gts_eval.extend(labels_target)
			
			preds_eval = np.asarray(preds_eval)
			gts_eval = np.asarray(gts_eval)

			score_cls_val = (np.mean(preds_eval == gts_eval)).astype(np.float)
			print('\n({}) acc. v {:.4f}\n'.format(epoch, 100*score_cls_val))

	
	writer_tensorboard.add_scalar('reconstruction_s loss', np.mean(recon_losses_s), epoch)
	writer_tensorboard.add_scalar('loglikelihood_s loss', np.mean(like_losses_s), epoch)
	writer_tensorboard.add_scalar('entropy loss', np.mean(entr_L), epoch)
	writer_tensorboard.add_scalar('log_s det', np.mean(log_DS), epoch)
	writer_tensorboard.add_scalar('source_pred loss', np.mean(classifier_losses), epoch)
	writer_tensorboard.add_scalar('reconstruction_t loss', np.mean(recon_losses_t), epoch)
	writer_tensorboard.add_scalar('loglikelihood_t loss', np.mean(like_losses_t), epoch)
	writer_tensorboard.add_scalar('log_t det', np.mean(log_DT), epoch)
	writer_tensorboard.add_scalar('cond_entropy loss', np.mean(conent_losses), epoch)
	writer_tensorboard.add_scalar('trg_val_accuracy', 100*score_cls_val, epoch)
	writer_tensorboard.add_scalar('acc_cls_src', 100*acc_cls_src, epoch)


	if epoch % 150 ==0:
		torch.save(encoder_net.state_dict(), os.path.join(args.savedir, 'ENCModel_epo{}'.format(epoch)))
		torch.save(fclayer_snet.state_dict(), os.path.join(args.savedir, 'FCModel_s_epo{}'.format(epoch)))
		torch.save(decoder_snet.state_dict(), os.path.join(args.savedir, 'DECModel_s_epo{}'.format(epoch)))
		torch.save(flow_snet.state_dict(), os.path.join(args.savedir, 'FLOWModel_s_epo{}'.format(epoch)))

		torch.save(fclayer_snet.state_dict(), os.path.join(args.savedir, 'FCModel_t_epo{}'.format(epoch)))
		torch.save(decoder_tnet.state_dict(), os.path.join(args.savedir, 'DECModel_t_epo{}'.format(epoch)))
		torch.save(flow_tnet.state_dict(), os.path.join(args.savedir, 'FLOWModel_t_epo{}'.format(epoch)))
		torch.save(classifier_net.state_dict(), os.path.join(args.savedir, 'CLModel_epo{}'.format(epoch)))
