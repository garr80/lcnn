import lmdb
import torch
import random
import numpy as np
from collections import defaultdict
from definition_pb2 import Datum
import threading 
import time
from librosa.effects import trim
# from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader,Dataset
import librosa
def extract_logcqt(utt,n_bins=28):
    bins_per_octave = 12
    hop_length = 512
    fmin = 64
    cqt = np.abs( librosa.cqt(utt,sr=16000,hop_length=hop_length,fmin=fmin,bins_per_octave=bins_per_octave,
    n_bins=n_bins))
    logcqt = librosa.core.power_to_db(cqt*cqt,top_db=None )
    return logcqt.T



class LMDBParser(Dataset):
    def __init__(self,lmdb_file,key_file,tf_mask=False ,max_frame=750,F_max_mask=12,frame_dim=28):
        self.lmdb_file = lmdb_file
        self.key_file = key_file
        self.read_keys()
        self.frame_dim=frame_dim
        self.len=len(self.keys)
        self.tf_mask = tf_mask
        self.max_frame=max_frame
        self._lmdb_env = lmdb.open(self.lmdb_file,readonly=True,lock=False)
        self.F_max_mask=F_max_mask
        # self._1mdb_cursor = self._1mdb_env.begin().cursor() 
    def tf_masking(self,length,dim):
        max_mask = self.F_max_mask
        mask = random.randint(0,F_max_mask)
        F0_mask = random.randint(0,dim-F_mask)
        data_tf_mask = np.ones((length,dim),dtype=' float32' )
        # data tf_mask[T0_mask:T0_mask+T_mask,FO mask:F0_mask+F_mask] = 0.0
        data_tf_mask[0:length,F0_mask:F0_mask+F_mask] = 0.0
        return data_tf_mask

    def read_keys(self):
        self.keys=[]
        with open(self.key_file)as fin:
            for line in fin:
                self.keys.append(line.strip().encode())

    def __getitem__(self,index):
        datum = Datum( )
        with self._lmdb_env.begin(write=False) as txn:
            #print(self.keys[index])
            value = txn.get(b" "+self.keys[index])
        #print(value)    
        datum.ParseFromString(value )
        feature_frames = int (datum.frame_length)
        #frame_dim = int(datum.feature_dim)
        feature = np.frombuffer(datum.data,dtype = np.float32)
        #print( "before",feature.shape )
        if 1 :
            _,index=trim( feature,top_db=60 ,ref=40,frame_length=512 ,hop_length=128)
            feature=feature [ index[0]: index[1]] 
        #print(feature.shape)
        #print(feature.shape)
        feature=extract_logcqt(feature ,self.frame_dim)
        #print(feature.shape)
        if feature.shape[0] < self.max_frame:
            feature = np.tile(feature,(self.max_frame // feature.shape[0] + 1,1))
        feature= feature[ :self.max_frame,: ]
        #print(feature.shape)
        #feature.setflags (write=1)
        if self.tf_mask:
            data_tf_mask =self.tf_masking(self.max_frame,self.frame_dim)
            feature = feature * data_tf_mask
        #features = np.zeros((self.max_frames,self.feature_dim) ,dtype= 'float32' )
        label = datum.speaker_label
        wave_name = datum.wave_name.decode( )
        # feature = torch.Tensor( feature).to(torch.float32) 
        # label = torch.Tensor( label).long()
        return feature ,label ,wave_name
    def __len__(self):
        return self.len
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator (super (DataLoaderX,self).__iter__() ,max_prefetch=1)

def train_data_sampler(lmdb_file,key_file,batch_size,tf_mask ,threads ,max_frame,frame_dim):
    lmdb_parser = LMDBParser(lmdb_file,key_file,tf_mask ,max_frame,frame_dim=frame_dim)
    batch_data = DataLoader( dataset= lmdb_parser ,batch_size=batch_size,num_workers=threads )
    return batch_data

def setup_seed(random_seed,cudnn_deterministic=True):
    import os
    torch.manual_seed(random_seed )
    random.seed( random_seed)
    np.random.seed(random_seed)
    os.environ[ ' PYTHONHASHSEED'] = str(random_seed)
    if torch.cuda.is_available( ):
        torch.cuda.manual_seed_all( random_seed )
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

if __name__=="__main__":
    lmdb_file="/home/lhj/datasource/lmdb/train/raw_no_v0/lmdb_data"
    key_file="/home/lhj/datasource/lmdb/train/raw_no_v0/keys/total.key"
    setup_seed(7)
    batch_size=2
    start=time. time( )
    a=train_data_sampler(lmdb_file,key_file, batch_size,tf_mask=False, threads=20, max_frame=282 ,frame_dim=56)
    import time
    #start=time. time( )
    for num,i in enumerate(a):
        if num>1000:
            break
        # a,b,wave_name=i 
        # print (num,wave_name )
        # print(a. shape )
        # if num>10:
        #     break
    print(time. time()-start)
