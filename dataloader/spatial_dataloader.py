#coding=utf-8
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure

'''
Dataset - DataLoader
'''
class spatial_dataset(Dataset):  
    '''
    继承自 pytorch 的 DataSet类, 存储和读取图片以及标签信息
    '''
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()    #视频名称
        self.values=dic.values()  #视频对用的标签
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)
    '''
    输入  [视频名，索引] 表示载入某个视频的某一帧
    '''
    def load_ucf_image(self,video_name, index):
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandStandPushups_'+g
            path = self.root_dir + 'HandstandPushups'+'/v_'+name+'/'
        else:
            path = self.root_dir + video_name.split('_')[0]+'/v_'+video_name+'/'
        img = Image.open(path + 'image_{:05d}.jpg'.format(index))
        transformed_img = self.transform(img)
        img.close()

        return transformed_img
   

    def __getitem__(self, idx):

        '''
        idx -> 对应一个视频
        此函数为 从一个视频中随机选取 3 帧 作为训练数据

        训练时: 对应随机生成三个数
        例如：  10 对应的是 clips: [5,6,9]

        返回
        train: sample  ({img1:图像数组1，img2:图像数组2，img3:图像数组3}，标签]
        val  : sample  (video_name, 图像数组，标签)
        '''
        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips/3))
            clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            
        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i] #视频中对应3帧的索引
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

#---------------------------------------------分界线-----------------------------------------
class spatial_dataloader():
    '''
    运行run方法
       (1) load_frame_count 获得对应视频的帧数                 ->   frame_count[视频名（与trainlist 和 testlist的形式一致）]= 帧数
       (2) get_training_dic 将对应视频帧数(-10+1) 加到视频名后 ->   dic_training[视频名_帧数] = 标签
       (3) val_sample20     对于测试样例 每个视频均匀生成19张  ->   dic_testing [视频名_index]= 标签
       (4) train    -> train_loader (包含transform)
       (5) validate -> val_loader   (包含transform)
    
    返回 train_loader 和 test_loader
    '''
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        '''
         获得train和 test的字典
        '''
        self.train_video, self.test_video = splitter.split_video()

    '''
    获得对应视频的帧数 frames
    frame_count[视频名（与trainlist 和 testlist的形式一致）]= 帧数
    '''
    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open('/home/chao/chao/action/pytorch/two-stream/dataloader/dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
	    	print videoname
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    '''
    确定训练使用frame， 加在视频名之后
    '''
    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]

    '''
    确定测试使用的frames， 加在视频名之后
    每个视频选取19张测试图片
    '''                    
    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'
        print training_set[1][0]['img1'].size()
         
        print ('\n')    
        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),#Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

'''
if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/home/chao/dataset/ucf-101-jpg/', 
                                ucf_list='/home/chao/chao/action/pytorch/two-stream/UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()
'''
