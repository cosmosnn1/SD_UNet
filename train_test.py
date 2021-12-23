"""Train/Test"""
"""SD-UNET"""
""": A Novel Segmentation Framework for CT Images of 3 Lung Infections"""

import os
import numpy as np
import cv2
from PIL import Image
import time
import datetime
import torch
import torchvision
import numpy
from torch import optim
from torch.autograd import Variable
from utils import helpers
import torch  as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,CE_Net_,NestedUNet,init_weights
from unet.unet_model import DualNorm_Unet
from fcn8s import FCN8
import csv
from Loss import lossfunction,lossfunction_3
"""train/test"""
class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.imgsize=config.image_size
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob
		#segmentation class
		self.classes=config.classes

		# Hyper-parameters
		self.lr = config.lr
		#self.beta1 = config.beta1
		#self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		if config.classes==1:

			self.model_path = config.model_path
			self.result_path = config.result_path
		if config.classes > 1:

			self.model_path = config.model_path_3
			self.result_path = config.result_path_3


		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

		#self.unet_path = os.path.join(self.model_path,'classnumber=%d-%s-%d-%.4f-%d-%.4f.pkl' % (self.classes,self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			if self.classes==1:
				self.unet = U_Net(img_ch=3, output_ch=1)
				net = U_Net()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init
			if self.classes >1:
				self.unet = U_Net(img_ch=3, output_ch=3)
				net = U_Net()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init

		if self.model_type =='CE_Net_':
			if self.classes==1:
				self.unet = CE_Net_(num_channels=3, num_classes=1)
				net = FCN8()  # generate an instance network from the Net class
				#net.apply(init_weights)  # apply weight init
			if self.classes >1:
				self.unet = CE_Net_(img_ch=3, output_ch=3)
				net = CE_Net_()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init
		if self.model_type =='NestedUNet':
			if self.classes==1:
				self.unet = NestedUNet(args=0, in_channel=3, out_channel=1)
				net = FCN8()  # generate an instance network from the Net class
				#net.apply(init_weights)  # apply weight init
			if self.classes >1:
				self.unet = NestedUNet(in_channel=3, out_channel=3)
				net = CE_Net_()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init

		elif self.model_type =='R2U_Net':
			if self.classes==1:
				self.unet =R2U_Net(img_ch=3,output_ch=1,t=self.t)

			if self.classes >1:
				self.unet = R2U_Net(img_ch=3,output_ch=3,t=self.t)

		elif self.model_type =='AttU_Net':
			if self.classes==1:
				self.unet = AttU_Net(img_ch=3,output_ch=1)
			if self.classes >1:
				self.unet = AttU_Net(img_ch=3,output_ch=3)

		elif self.model_type == 'R2AttU_Net':

			if self.classes==1:
				self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			if self.classes >1:
				self.unet = R2AttU_Net(img_ch=3,output_ch=3,t=self.t)

		elif self.model_type == 'DualNorm_Unet':
			if self.classes==1:
				self.unet = DualNorm_Unet(img_ch=3, output_ch=1)
				net = DualNorm_Unet()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init
			if self.classes >1:
				self.unet =DualNorm_Unet(img_ch=3, output_ch=3)
				net = DualNorm_Unet()  # generate an instance network from the Net class
				net.apply(init_weights)  # apply weight init

		# if self.classes == 1:
		# 	self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, weight_decay=0.0005)  # 权重衰减设置，即l2正则化
		# if self.classes > 1:
		# 	self.optimizer = optim.SGD(list(self.unet.parameters()), self.lr, momentum=0.9, weight_decay=0.0005)
		self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr)  #, weight_decay=0.000

		self.unet.to(self.device)
		# self.print_network(self.unet, self.model_type)
"""train"""
	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		global acc, SE, SP, PC, F1, DC, JS, loss, epoch

		# SD-UNet Train
		if os.path.isfile(self.unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(self.unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,self.unet_path))
		else:
			
			lr = self.lr
			best_unet_score = 0.
			#with torch.no_grad():
			best_epoch=0
			print(self.classes)
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				acc1 = 0.  # Accuracy
				SE1 = 0.  # Sensitivity (Recall)
				SP1 = 0.  # Specificity
				PC1 = 0.  # Precision
				F11 = 0.  # F1 Score
				JS1 = 0.  # Jaccard Similarity
				DC1 = 0.  # Dice Coefficient

				acc2 = 0.  # Accuracy
				SE2 = 0.  # Sensitivity (Recall)
				SP2 = 0.  # Specificity
				PC2 = 0.  # Precision
				F12 = 0.  # F1 Score
				JS2 = 0.  # Jaccard Similarity
				DC2 = 0.  # Dice Coefficient
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth
					images = images.to(self.device)
					GT = GT.to(self.device)


					# SR : Segmentation Result


					SR = self.unet(images)


					# spade result
					SR_probs = F.sigmoid(SR)
					if self.classes == 1:
						SR_flat = SR_probs.view(SR_probs.size(0), -1)
						GT_flat = GT.view(GT.size(0), -1)
						loss = lossfunction(SR_flat, GT_flat)
					if self.classes > 1:
						loss = lossfunction_3(SR, GT)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()
					# evaluation

					if self.classes==1:
						acc += get_accuracy(SR, GT)
						SE += get_sensitivity(SR, GT)
						SP += get_specificity(SR, GT)
						PC += get_precision(SR, GT)
						F1 += get_F1(SR, GT)
						JS += get_JS(SR, GT)
						DC += get_DC(SR, GT)

					if self.classes>1:
						#MASK PIXEL=127 GGO
						acc1 += get_accuracy(SR[:,1:2,:], GT[:,1:2,:])
						SE1 += get_sensitivity(SR[:,1:2,:], GT[:,1:2,:])
						SP1 += get_specificity(SR[:,1:2,:], GT[:,1:2,:])
						PC1 += get_precision(SR[:,1:2,:], GT[:,1:2,:])
						F11 += get_F1(SR[:,1:2,:], GT[:,1:2,:])
						JS1 += get_JS(SR[:,1:2,:], GT[:,1:2,:])
						DC1 += get_DC(SR[:,1:2,:], GT[:,1:2,:])
						# MASK PIXEL=255 consolidation
						acc2 += get_accuracy(SR[:, 2:3, :], GT[:, 2:3, :])
						SE2 += get_sensitivity(SR[:, 2:3, :], GT[:, 2:3, :])
						SP2 += get_specificity(SR[:, 2:3, :], GT[:, 2:3, :])
						PC2 += get_precision(SR[:, 2:3, :], GT[:, 2:3, :])
						F12 += get_F1(SR[:, 2:3, :], GT[:, 2:3, :])
						JS2 += get_JS(SR[:, 2:3, :], GT[:, 2:3, :])
						DC2 += get_DC(SR[:, 2:3, :], GT[:, 2:3, :])

						acc = (acc1 + acc2) / 2  # Accuracy
						SE = (SE1 + SE2) / 2   # Sensitivity (Recall)
						SP = (SP1+SP2)/2  # Specificity
						PC = (PC1+PC2)/2  # Precision
						F1 = (F11+F12)/2 # F1 Score
						JS = (JS1+JS2)/2  # Jaccard Similarity
						DC = (DC1+DC2)/2  # Dice Coefficient
					length += 1  # images.size(0)
				if self.classes == 1:
					acc = acc / length
					SE = SE / length
					SP = SP / length
					PC = PC / length
					F1 = F1 / length
					JS = JS / length
					DC = DC / length

				if self.classes > 1:
					acc1 = acc1 / length
					SE1 = SE1 / length
					SP1 = SP1 / length
					PC1 = PC1 / length
					F11 = F11 / length
					JS1 = JS1 / length
					DC1 = DC1 / length

					acc2 = acc2 / length
					SE2 = SE2 / length
					SP2 = SP2 / length
					PC2 = PC2 / length
					F12 = F12 / length
					JS2 = JS2 / length
					DC2 = DC2 / length

					acc = acc / length
					SE = SE / length
					SP = SP / length
					PC = PC / length
					F1 = F1 / length
					JS = JS / length
					DC = DC / length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] DC: %.4f\n,JS: %.4f,Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f ' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  DC,JS,acc,SE,SP,PC,F1))
				if self.classes>1:
					print('Training] DC1: %.4f\n,JS1: %.4f,Acc1: %.4f, SE1: %.4f, SP1: %.4f, PC1: %.4f, F11: %.4f ' % (
							DC1, JS1, acc1, SE1, SP1, PC1, F11))
					print(' [Training] DC2: %.4f\n,JS2: %.4f,Acc2: %.4f, SE2: %.4f, SP2: %.4f, PC2: %.4f, F12: %.4f ' % (
							DC2, JS2, acc2, SE2, SP2, PC2, F12))
				#
				# # Decay learning rate
				# #if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				# if self.classes == 1:
				# 	if (epoch + 1) % 10 == 0:  # 每10个epoch调整一次学习率
				# 		lr = lr * 0.8
				# 		# lr -= (self.lr / float(self.num_epochs_decay))
				# 		# 固定步长衰减
				# 		# optimizer_StepLR = torch.optim.SGD(net.parameters(), lr=0.1)
				# 		# StepLR = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=step_size, gamma=0.65)
				# 		for param_group in self.optimizer.param_groups:
				# 			param_group['lr'] = lr
				# 		print('Decay learning rate to lr: {}.'.format(lr))
				#
				# if self.classes > 1:
				# 	if (epoch + 1) % 10 == 0:  # 每10个epoch调整一次学习率
				# 		lr = lr * 0.5
				# 		# lr -= (self.lr / float(self.num_epochs_decay))
				# 		# 固定步长衰减
				# 		# optimizer_StepLR = torch.optim.SGD(net.parameters(), lr=0.1)
				# 		# StepLR = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=step_size, gamma=0.65)
				# 		for param_group in self.optimizer.param_groups:
				# 			param_group['lr'] = lr
				# 		print('Decay learning rate to lr: {}.'.format(lr))
				if (epoch + 1) % 10 == 0:
					lr=lr * 0.8
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print('Decay learning rate to lr: {}.'.format(lr))




				# if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
				# 	lr -= (self.lr / float(self.num_epochs_decay))
				# 	for param_group in self.optimizer.param_groups:
				# 		param_group['lr'] = lr
				# 	print('Decay learning rate to lr: {}.'.format(lr))




				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc1 = 0.  # Accuracy
				SE1 = 0.  # Sensitivity (Recall)
				SP1 = 0.  # Specificity
				PC1 = 0.  # Precision
				F11 = 0.  # F1 Score
				JS1 = 0.  # Jaccard Similarity
				DC1 = 0.  # Dice Coefficient

				acc2 = 0.  # Accuracy
				SE2 = 0.  # Sensitivity (Recall)
				SP2 = 0.  # Specificity
				PC2 = 0.  # Precision
				F12 = 0.  # F1 Score
				JS2 = 0.  # Jaccard Similarity
				DC2 = 0.  # Dice Coefficient

				acc = 0.  # Accuracy
				SE = 0.  # Sensitivity (Recall)
				SP = 0.  # Specificity
				PC = 0.  # Precision
				F1 = 0.  # F1 Score
				JS = 0.  # Jaccard Similarity
				DC = 0.  # Dice Coefficient

				length=0
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					# images=Variable(images, requires_grad=True)
					# GT = Variable(GT, requires_grad=True)
					with torch.no_grad():
						SR = F.sigmoid(self.unet(images))

					#evaluation
					if self.classes == 1:
						acc += get_accuracy(SR, GT)
						SE += get_sensitivity(SR, GT)
						SP += get_specificity(SR, GT)
						PC += get_precision(SR, GT)
						F1 += get_F1(SR, GT)
						JS += get_JS(SR, GT)
						DC += get_DC(SR, GT)

					if self.classes > 1:
						# MASK PIXEL=127 GGO
						acc1 += get_accuracy(SR[:, 1:2, :], GT[:, 1:2, :])
						SE1 += get_sensitivity(SR[:, 1:2, :], GT[:, 1:2, :])
						SP1 += get_specificity(SR[:, 1:2, :], GT[:, 1:2, :])
						PC1 += get_precision(SR[:, 1:2, :], GT[:, 1:2, :])
						F11 += get_F1(SR[:, 1:2, :], GT[:, 1:2, :])
						JS1 += get_JS(SR[:, 1:2, :], GT[:, 1:2, :])
						DC1 += get_DC(SR[:, 1:2, :], GT[:, 1:2, :])
						# MASK PIXEL=255 consolidation
						acc2 += get_accuracy(SR[:, 2:3, :], GT[:, 2:3, :])
						SE2 += get_sensitivity(SR[:, 2:3, :], GT[:, 2:3, :])
						SP2 += get_specificity(SR[:, 2:3, :], GT[:, 2:3, :])
						PC2 += get_precision(SR[:, 2:3, :], GT[:, 2:3, :])
						F12 += get_F1(SR[:, 2:3, :], GT[:, 2:3, :])
						JS2 += get_JS(SR[:, 2:3, :], GT[:, 2:3, :])
						DC2 += get_DC(SR[:, 2:3, :], GT[:, 2:3, :])

						acc = (acc1 + acc2) / 2  # Accuracy
						SE = (SE1 + SE2) / 2  # Sensitivity (Recall)
						SP = (SP1 + SP2) / 2   #Specificity
						PC = (PC1 + PC2) / 2  # Precision
						F1 = (F11 + F12) / 2  # F1 Score
						JS = (JS1 + JS2) / 2  # Jaccard Similarity
						DC = (DC1 + DC2) / 2  # Dice Coefficient

					length += 1                  #  images.size(0)

				if self.classes == 1:
					acc = acc / length
					SE = SE / length
					SP = SP / length
					PC = PC / length
					F1 = F1 / length
					JS = JS / length
					DC = DC / length

				if self.classes > 1:
					acc1 = acc1 / length
					SE1 = SE1 / length
					SP1 = SP1 / length
					PC1 = PC1 / length
					F11 = F11 / length
					JS1 = JS1 / length
					DC1 = DC1 / length

					acc2 = acc2 / length
					SE2 = SE2 / length
					SP2 = SP2 / length
					PC2 = PC2 / length
					F12 = F12 / length
					JS2 = JS2 / length
					DC2 = DC2 / length

					acc = acc / length
					SE = SE / length
					SP = SP / length
					PC = PC / length
					F1 = F1 / length
					JS = JS / length
					DC = DC / length
				unet_score = JS + DC

				print('[Validation] DC: %.4f\n, JS: %.4f,Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f '%(DC,JS,acc,SE,SP,PC,F1))
				if self.classes > 1:
					print(
						'[Validation_GGO] DC1: %.4f\n, JS1: %.4f,Acc1: %.4f, SE1: %.4f, SP1: %.4f, PC1: %.4f, F11: %.4f ' % (
							DC1, JS1, acc1, SE1, SP1, PC1, F11))
					print(
						'[Validation_con] DC2: %.4f\n, JS2: %.4f,Acc2: %.4f, SE2: %.4f, SP2: %.4f, PC2: %.4f, F12: %.4f ' % (
							DC2, JS2, acc2, SE2, SP2, PC2, F12))
				# Save U-Net model
				if (epoch + 1) % 10 == 0:
					best_unet = self.unet.state_dict()
					torch.save(best_unet, self.unet_path + "SA-UNet-%d" % (epoch + 1))


				#Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch + 1
					# best_unet=0
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
					torch.save(best_unet, self.unet_path)





			print('Best %s best_epoch = %d'%(self.model_type,best_epoch))
			#===================================== Test ====================================#
			f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(["train", DC, JS, self.model_type, acc, SE, SP, PC, F1, self.lr, best_epoch, self.num_epochs,
						 self.num_epochs_decay, self.augmentation_prob,self.imgsize])
			f.close()
			del self.unet
			del best_unet
"""test"""
	def test(self):


			self.build_model()
			self.unet.load_state_dict(torch.load(self.unet_path))
			
			self.unet.train(False)
			self.unet.eval()
			imgs = []
			preds=[]
			preds1=[]


			gts = []
			acc1 = 0.  # Accuracy
			SE1 = 0.  # Sensitivity (Recall)
			SP1 = 0.  # Specificity
			PC1 = 0.  # Precision
			F11 = 0.  # F1 Score
			JS1 = 0.  # Jaccard Similarity
			DC1 = 0.  # Dice Coefficient

			acc2 = 0.  # Accuracy
			SE2 = 0.  # Sensitivity (Recall)
			SP2 = 0.  # Specificity
			PC2 = 0.  # Precision
			F12 = 0.  # F1 Score
			JS2 = 0.  # Jaccard Similarity
			DC2 = 0.  # Dice Coefficient

			acc = 0.  # Accuracy
			SE = 0.  # Sensitivity (Recall)
			SP = 0.  # Specificity
			PC = 0.  # Precision
			F1 = 0.  # F1 Score
			JS = 0.  # Jaccard Similarity
			DC = 0.  # Dice Coefficient
			length=0
			_j = 0
			for i, (images, GT) in enumerate(self.valid_loader):

				images = images.to(self.device)
				#print(images.shape)
				GT = GT.to(self.device)

				with torch.no_grad():
					SR = F.sigmoid(self.unet(images))

					#print("Sssss123")
				#evaluation
				if self.classes == 1:
					acc += get_accuracy(SR, GT)
					SE += get_sensitivity(SR, GT)
					SP += get_specificity(SR, GT)
					PC += get_precision(SR, GT)
					F1 += get_F1(SR, GT)
					JS += get_JS(SR, GT)
					DC += get_DC(SR, GT)

				if self.classes > 1:
					# MASK PIXEL=127 GGO#索引大于等于1小于4的切片
					acc1 += get_accuracy(SR[:, 1:2, :], GT[:, 1:2, :])
					SE1 += get_sensitivity(SR[:, 1:2, :], GT[:, 1:2, :])
					SP1 += get_specificity(SR[:, 1:2, :], GT[:, 1:2, :])
					PC1 += get_precision(SR[:, 1:2, :], GT[:, 1:2, :])
					F11 += get_F1(SR[:, 1:2, :], GT[:, 1:2, :])
					JS1 += get_JS(SR[:, 1:2, :], GT[:, 1:2, :])
					DC1 += get_DC(SR[:, 1:2, :], GT[:, 1:2, :])
					# MASK PIXEL=255 consolidation
					acc2 += get_accuracy(SR[:, 2:3, :], GT[:, 2:3, :])
					SE2 += get_sensitivity(SR[:, 2:3, :], GT[:, 2:3, :])
					SP2 += get_specificity(SR[:, 2:3, :], GT[:, 2:3, :])
					PC2 += get_precision(SR[:, 2:3, :], GT[:, 2:3, :])
					F12 += get_F1(SR[:, 2:3, :], GT[:, 2:3, :])
					JS2 += get_JS(SR[:, 2:3, :], GT[:, 2:3, :])
					DC2 += get_DC(SR[:, 2:3, :], GT[:, 2:3, :])

					acc = (acc1 + acc2) / 2  # Accuracy
					SE = (SE1 + SE2) / 2  # Sensitivity (Recall)
					SP = (SP1 + SP2) / 2  # Specificity
					PC = (PC1 + PC2) / 2  # Precision
					F1 = (F11 + F12) / 2  # F1 Score
					JS = (JS1 + JS2) / 2  # Jaccard Similarity
					DC = (DC1 + DC2) / 2  # Dice Coefficient
						
				length +=1 #images.size(0)
				if self.classes > 1:

					palette = [[0], [127], [255]]

					GT = GT.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					GT = helpers.onehot_to_mask(GT, palette)
					SR1 = SR
					SR = SR.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					SR = helpers.onehot_to_mask(SR, palette)
					images = images.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
					images = np.expand_dims(images, axis=2)
					imgs.append(images * 255)
					preds.append(SR)
					gts.append(GT)

				if self.classes == 1:
					SR1=SR
					GT = GT.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					SR = SR.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					images = images.cuda().detach().cpu().numpy()[0].transpose([1, 2, 0])
					images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
					images = np.expand_dims(images, axis=2)
					imgs.append(images * 255)
					preds.append(SR*255)
					gts.append(GT*255)
					# save 2class prediction images

			if self.classes == 1:
				acc = acc / length
				SE = SE / length
				SP = SP / length
				PC = PC / length
				F1 = F1 / length
				JS = JS / length
				DC = DC / length

			if self.classes>1:
				acc1 = acc1 / length
				SE1 = SE1 / length
				SP1 = SP1 / length
				PC1 = PC1 / length
				F11 = F11 / length
				JS1 = JS1 / length
				DC1 = DC1 / length

				acc2 = acc2 / length
				SE2 = SE2 / length
				SP2 = SP2 / length
				PC2 = PC2 / length
				F12 = F12 / length
				JS2 = JS2 / length
				DC2 = DC2 / length

				acc = acc / length
				SE = SE / length
				SP = SP / length
				PC = PC / length
				F1 = F1 / length
				JS = JS / length
				DC = DC / length



			imgs = np.hstack([*imgs])
			preds = np.hstack([*preds])
			gts = np.hstack([*gts])
			show_res = np.vstack(np.uint8([imgs,preds, gts]))

			cv2.imshow("top is mri , middle is pred,  bottom is gt", show_res)

			cv2.waitKey(0)

			unet_score = JS + DC

				# 用来看预测的像素值
				# pred_showval = SR
				# pred = helpers.onehot_to_mask(pred, bladder.palette)
				# np.uint8()反归一化到[0, 255]

			print("unet_score_test= ", unet_score)
			print('\n[Testing] DC: %.4f\n,JS: %.4f,Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f ' %(
				DC,JS,acc, SE, SP, PC, F1))
			if self.classes == 1:
				f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(f)
				wr.writerow(["test", DC, JS, self.model_type, acc, SE, SP, PC, F1, self.lr, self.num_epochs,
							 self.num_epochs_decay, self.augmentation_prob])
				f.close()

			if self.classes > 1:
				print(
					'\n[Testing_GGO] DC1: %.4f\n,JS1: %.4f,Acc1: %.4f, SE1: %.4f, SP1: %.4f, PC1: %.4f, F11: %.4f ' % (
						DC1, JS1, acc1, SE1, SP1, PC1, F11))
				print(
					'\n[Testing_con] DC2: %.4f\n,JS2: %.4f,Acc2: %.4f, SE2: %.4f, SP2: %.4f, PC2: %.4f, F12: %.4f ' % (
						DC2, JS2, acc2, SE2, SP2, PC2, F12))
				f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(f)
				wr.writerow(["GGO","test", DC1, JS1, self.model_type, acc1, SE1, SP1, PC1, F11, "GGO",self.lr, self.num_epochs,
							 self.num_epochs_decay, self.augmentation_prob])
				wr.writerow(["CON","test", DC2, JS2, self.model_type, acc2, SE2, SP2, PC2, F12,"CON",self.lr, self.num_epochs,
							 self.num_epochs_decay, self.augmentation_prob])
				wr.writerow(["test", DC, JS, self.model_type, acc, SE, SP, PC, F1, self.lr, self.num_epochs,
							 self.num_epochs_decay, self.augmentation_prob],)

				f.close()

			
