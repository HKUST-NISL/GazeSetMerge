import os
import glob
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import vector_to_pitchyaw_one

def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]

    anno.subid = int(anno.face.split('/')[0][1:])
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]

    anno.subid = int(anno.face.split('/')[0][1:]) - 1
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    anno.subid = int(anno.face.split('/')[0][7:])
    return anno

def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping

def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]

def readfolder(data, specific=None, reverse=False):

    """" 
    Traverse the folder 'data.label' and read data from all files in the folder.
    
    Specific is a list, specify the num of extracted file.
    When reverse is True, read the files which num is not in specific. 
    """
 
    # folders = os.listdir(data.label)

    folders = glob.glob(data.label+'/*.label')
    ldm_folders = glob.glob(data.landmark+'/*.landmark')
    folders.sort()
    ldm_folders.sort()

    folder = folders
    ldm_folder = ldm_folders
    if specific is not None:
        if reverse:
            num = np.arange(len(folders))
            specific = list(filter(lambda x: x not in specific, num))
        
        folder = [folders[i] for i in specific]

    data.label = folder
    data.landmark = ldm_folder

    return data, folders
    

class trainloader(Dataset): 
  def __init__(self, dataset):

    # Read source data
    self.data = edict() 
    self.data.line = []
    self.data.landmark = None
    self.data.root = dataset.image
    self.data.decode = Get_Decode(dataset.name)
    self.data.type = source.name

    if dataset.isFolder:
        dataset, _ = readfolder(dataset)

    if isinstance(dataset.label, list):

      for i in dataset.label:

        with open(i) as f: line = f.readlines()

        if dataset.header: line.pop(0)

        self.data.line.extend(line)

    else:

      with open(dataset.label) as f: self.data.line = f.readlines()

      if dataset.header: self.data.line.pop(0)
    

    if isinstance(dataset.landmark, list):

      lmk_arrays = []
      for i in dataset.landmark:
        lmk_array = np.loadtxt(i)
        lmk_arrays.append(lmk_array)
      self.data.landmark = np.vstack(lmk_arrays)
    else:
      self.data.landmark = np.loadtxt(dataset.landmark)
    
    self.data.landmark = self.data.landmark.astype(np.int32)


    # build transforms
    self.transforms = transforms.Compose([
        transforms.ToTensor()
    ])


  def __len__(self):

    return len(self.data.line)


  def __getitem__(self, idx):

    # Read souce information
    line = self.data.line[idx]
    line = line.strip().split(" ")
    anno = self.data.decode(line)

    # img = cv2.imread(os.path.join(self.data.root, anno.face))
    # img = self.transforms(img)

    if self.data.type == 'eth':
        label = np.array(anno.gaze2d.split(",")).astype("float")
        head = np.array(anno.head2d.split(",")).astype("float")
    else:
        label3d = np.array(anno.gaze3d.split(",")).astype("float") * -1
        head3d = np.array(anno.head3d.split(",")).astype("float") * -1

        label = vector_to_pitchyaw_one(label3d)
        head = vector_to_pitchyaw_one(head3d)

    landmark = self.data.landmark[idx]

    data = edict()
    data.face = anno.face
    data.type = self.data.type
    data.name = anno.name
    data.subid = anno.subid
    data.gaze = label
    data.head = head
    data.landmark = landmark

    return data

def loader(source, batch_size, shuffle=True,  num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

def encode_data(data):

    dirs = {'eth': 'ETH-Gaze/Image/train', 'diap': 'EyeDiap/Image', 'mpii': 'MPIIFaceGaze/Image'}
    type = data.type
    gaze = data.gaze
    head = data.head
    lmk = data.landmark
    subid = data.subid

    face_path = os.path.join(dirs[type], data.face)

    line = '%s %d %s %f,%f %f,%f ' % (type, subid, face_path, gaze[0], gaze[1], head[0], head[1])

    lmk_line = ','.join(str(x) for x in lmk)

    line = line + lmk_line
    return line


if __name__ == '__main__':

    all_path = './data/GazeSetsPng/all.label'

    sources = []
    source = edict()
    source.image = "./data/GazeSetsPng/MPIIFaceGaze/Image"
    source.label = "./data/GazeSetsPng/MPIIFaceGaze/Label"
    source.landmark = "./data/GazeSetsPng/MPIIFaceGaze/Landmark"
    source.header = True
    source.name = 'mpii'
    source.isFolder = True
    sources.append(source)

    source = edict()
    source.image = "./data/GazeSetsPng/EyeDiap/Image"
    source.label = "./data/GazeSetsPng/EyeDiap/Label"
    source.landmark = "./data/GazeSetsPng/EyeDiap/Landmark"
    source.header = True
    source.name = 'diap'
    source.isFolder = True
    sources.append(source)

    source = edict()
    source.image = "./data/GazeSetsPng/ETH-Gaze/Image/train"
    source.label = "./data/GazeSetsPng/ETH-Gaze/Label/train.label"
    source.landmark = "./data/GazeSetsPng/ETH-Gaze/Landmark/train.landmark"
    source.header = True
    source.name = 'eth'
    source.isFolder = False
    sources.append(source)

    with open(all_path, 'w') as f:

        for source in sources:
            dataset = trainloader(source)

            print(source.name, len(dataset))


            this_path = os.path.join('./data/GazeSetsPng/', source.name+'.label')

            this_file = open(this_path, 'w')

            # for i, data in enumerate(dataset):
            for i in tqdm(range(len(dataset))):
                data = dataset[i]
                face = data.face
                # print(i, data.face.shape, data.gaze, data.head, data.subid, data.landmark.shape)

                line = encode_data(data)
                # print(line)
                f.write(line+'\n')
                this_file.write(line+'\n')
                # break
                # cv2.imshow("face", face)
                # cv2.waitKey(0)
            
            this_file.close()
            
        