# coding=utf-8
import os, pickle

'''

使用方法：
 1.生成对象--splitter
 2.调用 split_video()
 3.split_video处理流程：
   get_action_index  -->  file2dic  -->  nameHans
 
 返回train 和 test两个 dic: [视频名称（不带前缀v_,例如：ApplyEyeMakeup_g10_c03）] = label

'''

class UCF101_splitter():
    '''
    path: trainlist和testlist 的路径
    splilt: 01 02 03
    '''
    def __init__(self, path, split):
        self.path = path
        self.split = split

    '''
    建立 类别名与标签的对关系： action_label[类别]=标签
    '''
    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label

    '''
    建立 trainlist 和 testlist 两个字典
    '''
    def split_video(self):
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')'
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)
	#self.train_video = train_video
        #self.test_video  = test_video

        return self.train_video, self.test_video

    '''
    对trainlist 和 testlist 进行类别和标签信息抽取
    dic [视频名称（不带前缀v_,例如：ApplyEyeMakeup_g10_c03）] = label
    '''
    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
        return dic

    '''
    调整类别名字不同的个例
    '''
    def name_HandstandPushups(self,dic):
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
		#videoname=video
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2


#if __name__ == '__main__':
#
#    path = '../UCF_list/'
#    split = '01'
#    splitter = UCF101_splitter(path=path,split=split)
#    train_video,test_video = splitter.split_video()
#    print len(train_video),len(test_video)
#    print type(train_video)
