from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader
import sys
if __name__ == '__main__':
    # rate = 0.231 best performance
    rate = float(sys.argv[1])	
    rgb_preds='record/spatial/resnet101/spatial_video_preds.pickle'
    outl_preds = 'record/motion/resnet101/motion_video_preds.pickle'
    subs_preds = 'record/subs/resnet18/subs_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(outl_preds,'rb') as f:
        outl =pickle.load(f)
    f.close()
    with open(subs_preds,'rb') as f:
	subs=pickle.load(f)
    f.close()
    
    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path='/home/chao/dataset/ucf-101-jpg/', 
                                    ucf_list='/home/chao/chao/action/pytorch/two-stream-action-recognition/UCF_list/',
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),101))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = outl[name]
	s = subs[name]
	#rate = 0.92
	#fusion = (0.769*r+0.231*s)*rate+(1-rate)*o
	#fusion =  r*rate+(1-rate)*o
        #fusion = 0.5*r+0.5*o
	fusion = r
	label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (fusion)
	#video_level_preds[ii,:].shape = 101
	#print ii
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(fusion) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                                
    print top1,top5
