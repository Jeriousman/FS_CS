from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import os
import cv2
import tqdm
import sys
sys.path.append('..')
# from utils.cap_aug import CAP_AUG

#/datasets/FFHQ
#data_path='/datasets/FFHQ'

class FaceEmbedCustom(TensorDataset):
    def __init__(self, data_path, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        # for data_path in data_path_list:
        image_list = glob.glob(f'{data_path}/*.*g')
        datasets.append(image_list)
        self.N.append(len(image_list))
        # with open(f'{data_path}/embed.pkl', 'rb') as f:
        #     embed = pickle.load(f)
        #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_mae = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
            
        image_path = self.datasets[idx][item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_mae(Xs), self.transforms_base(Xs), self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)

# dataset = FaceEmbedCustom('/workspace/examples/images/training')
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)




class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
            
        image_path = self.datasets[idx][item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)


#data_path='/datasets/FFHQ'
class FaceEmbedCelebA(TensorDataset):
    '''
    data_path안에 각 인물별로 폴더가 따로 존재하지 않고 모든이미지가 바로 폴더 하위에 저장되 있을때는 이 클래스를 쓴다. FFHQ, CelebA가 여기에 속한다
    '''
    def __init__(self, data_path, same_prob=0.8):
        self.data_path = data_path
        self.same_prob = same_prob
        datasets = glob.glob(f'{data_path}/*.*g', recursive=True)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        # idx = 0
        # while item >= self.N[idx]:
        #     item -= self.N[idx]
        #     idx += 1

        image_path = self.datasets[item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:  ##모든 데이터를 다 reconstruction loss를 계산하지 않고 어느정도의 확률로만 계산하기 위한 것
            # image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            image_path = self.datasets[random.randint(0, len(self.datasets)-1)]  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()  ##확률적으로 가끔은 같은 Xs와 같은 이미지로 Xt를 사용해서 reconstruction loss 를 계산하기 위함 
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)



#data_path='/datasets/FFHQ'
class FaceEmbedFFHQ(TensorDataset):
    '''
    data_path안에 각 인물별로 폴더가 따로 존재하지 않고 모든이미지가 바로 폴더 하위에 저장되 있을때는 이 클래스를 쓴다. FFHQ, CelebA가 여기에 속한다
    '''
    def __init__(self, data_path, same_prob=0.8):
        self.data_path = data_path
        self.same_prob = same_prob
        datasets = glob.glob(f'{data_path}/**/*.*g', recursive=True)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        # idx = 0
        # while item >= self.N[idx]:
        #     item -= self.N[idx]
        #     idx += 1

        image_path = self.datasets[item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:  ##모든 데이터를 다 reconstruction loss를 계산하지 않고 어느정도의 확률로만 계산하기 위한 것
            # image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            image_path = self.datasets[random.randint(0, len(self.datasets)-1)]  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()  ##확률적으로 가끔은 같은 Xs와 같은 이미지로 Xt를 사용해서 reconstruction loss 를 계산하기 위함 
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)




##/datasets/DOB/imagefile/DOB04F118SEC
class FaceEmbedSubdir(TensorDataset):
    '''
    VGG와 DOB데이터처럼 안에 각 인물마다 디렉토리가 따로 있는 경우 이 클래스를 사용한다
    '''
    
    def __init__(self, data_path, same_prob=0.8, same_identity=False, dataname='vgg'):

        self.same_prob = same_prob
        self.same_identity = same_identity
        self.dataname = dataname
        
        if self.dataname=='vgg':
                
            self.images_list = glob.glob(f'{data_path}/*/*.*g')
            self.folders_list = glob.glob(f'{data_path}/*')
            
            self.folder2imgs = {}

            for folder in tqdm.tqdm(self.folders_list):
                folder_imgs = glob.glob(f'{folder}/*')
                self.folder2imgs[folder] = folder_imgs
             
        if self.dataname=='dob':
            '''
            디오비 데이터를 모두 로딩하기 위해 걸린시간: 1930.1818947792053초
            '''
            self.images_list = glob.glob(f'{data_path}/*/*/*.*g') ##모든이미지의 full image list 
            self.folders_list = glob.glob(f'{data_path}/*/*')  ##  '/datasets/DOB/imagefile/DOB04F215SEC', '/datasets/DOB/imagefile/KTT03M008FIR', ...
            self.folder2imgs = {}

            for folder in tqdm.tqdm(self.folders_list): ##folder는 이미지를 가지고 있는 각각 폴더명
                folder_imgs = glob.glob(f'{folder}/*')  ##폴더명 아래에 있는 모든 파일을 가지고 와서 리스팅
                self.folder2imgs[folder] = folder_imgs ## 모든 이미지를 폴더경로 key에 저장하는 dict 생성. key의 예: folder2imgs['/datasets/DOB/imagefile2/CDW04F003FIR']
                
        
        
        self.N = len(self.images_list)
        
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.RandomHorizontalFlip(p=0.4),  ##Hojun added
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
            
        image_path = self.images_list[item]

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])

        if random.random() > self.same_prob:
            image_path = random.choice(self.images_list)
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return self.N
    
def MergeDict(dict1, dict2): 
    res = dict1 | dict2
    return res 


#data_path='/datasets/FFHQ'
##/datasets/DOB/imagefile/DOB04F118SEC
class FaceEmbedCombined_real(TensorDataset):
    '''
    CelebA나 FFHQ중 하나는 무조건 사용되어야 하게 설정되어있다.
    '''
    def __init__(self, vgg_data_path=None, celeba_data_path=None, dob_data_path=None, ffhq_data_path=None, same_prob=0.8, same_identity=False):

        self.vgg_data_path = vgg_data_path
        self.dob_data_path = dob_data_path
        self.ffhq_data_path = ffhq_data_path
        self.celeba_data_path = celeba_data_path
        self.total_dataset = []
        
        self.same_prob = same_prob
        self.same_identity = same_identity
        
        ##data_path='/datasets/FFHQ'
        if bool(self.ffhq_data_path)==True:
            self.ffhq_dataset = glob.glob(f'{self.ffhq_data_path}/**/*.*g', recursive=True)
            self.ffhq_folders = glob.glob(f'{self.ffhq_data_path}/*', recursive=True)
            self.ffhq_len = len(self.ffhq_dataset)

        ##'/datasets/CelebHQ/CelebA-HQ-img'
        if bool(self.celeba_data_path)==True:
            self.celeba_dataset = glob.glob(f'{self.celeba_data_path}/**.*g', recursive=True)
            self.celeba_len = len(self.celeba_dataset)
        # ffhq_folders = glob.glob(f'{self.ffhq_data_path}/*', recursive=True)
        
        
        if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            self.total_dataset = self.ffhq_dataset + self.celeba_dataset 

        elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
            self.total_dataset += self.celeba_dataset 
        
        elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
            self.total_dataset += self.ffhq_data_path 
            
        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')

        self.folder2imgs = {}
        
        if bool(self.vgg_data_path)==True:
                    
            self.vgg_dataset = glob.glob(f'{self.vgg_data_path}/*/*.*g')
            self.vgg_folders_list = glob.glob(f'{self.vgg_data_path}/*')
            


            for folder in tqdm.tqdm(self.vgg_folders_list):
                folder_imgs = glob.glob(f'{folder}/*')
                self.folder2imgs[folder] = folder_imgs
          
            self.vgg_len = len(self.vgg_dataset)
            self.total_dataset += self.vgg_dataset
               
        if bool(self.dob_data_path)==True:
            '''
            디오비 데이터를 모두 로딩하기 위해 걸린시간: 1930.1818947792053초
            '''
            self.dob_dataset = glob.glob(f'{self.dob_data_path}/*/*/*.*g') ##모든이미지의 full image list 
            self.dob_folders_list = glob.glob(f'{self.dob_data_path}/*/*')  ##  '/datasets/DOB/imagefile/DOB04F215SEC', '/datasets/DOB/imagefile/KTT03M008FIR', ...
            # self.dob_folder2imgs = {}

            for folder in tqdm.tqdm(self.dob_folders_list): ##folder는 이미지를 가지고 있는 각각 폴더명
                folder_imgs = glob.glob(f'{folder}/*')  ##폴더명 아래에 있는 모든 파일을 가지고 와서 리스팅
                self.folder2imgs[folder] = folder_imgs ## 모든 이미지를 폴더경로 key에 저장하는 dict 생성. key의 예: folder2imgs['/datasets/DOB/imagefile2/CDW04F003FIR']
                
            self.dob_len = len(self.dob_dataset)
            self.total_dataset += self.dob_dataset
    
        self.total_dataset_len = len(self.total_dataset)
        
        
        
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.RandomHorizontalFlip(p=0.4),  ##Hojun added
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):

        if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            if idx < self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_ffhq_celeba(image_path)
                
            elif idx >= self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_vgg_dob(image_path)
                
        # elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
        #     if idx < self.celeba_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_ffhq_celeba(image_path)
        #     elif idx >= self.celeba_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_vgg_dob(image_path)
                
        # elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
        #     if idx < self.ffhq_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_ffhq_celeba(image_path)
        #     elif idx >= self.ffhq_len:
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_vgg_dob(image_path)
        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')        

        return transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person

    def __len__(self):
        return self.total_dataset_len

    def fetch_ffhq_celeba(self, image_path):
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:  ##모든 데이터를 다 reconstruction loss를 계산하지 않고 어느정도의 확률로만 계산하기 위한 것
            # image_path = random.choice(self.total_dataset[random.randint(0, len(self.total_dataset)-1)])  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            image_path = self.total_dataset[random.randint(0, len(self.total_dataset)-1)]  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()  ##확률적으로 가끔은 같은 Xs와 같은 이미지로 Xt를 사용해서 reconstruction loss 를 계산하기 위함 
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def fetch_vgg_dob(self, image_path):

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])

        if random.random() > self.same_prob:
            if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len+self.celeba_len-1, self.total_dataset_len-1)]

            elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
                image_path = self.total_dataset[random.randint(self.celeba_len-1, self.total_dataset_len-1)]
            
            elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len-1, self.total_dataset_len-1)]
                
            else:
                raise ValueError('At least either CelebA and FFHQ data must be used')
            
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person











class FaceEmbedCombined_(TensorDataset):
    '''
    CelebA나 FFHQ중 하나는 무조건 사용되어야 하게 설정되어있다.
    '''
    def __init__(self, vgg_data_path=None, celeba_data_path=None, dob_data_path=None, ffhq_data_path=None, same_prob=0.8, same_identity=False):

        self.vgg_data_path = vgg_data_path
        self.dob_data_path = dob_data_path
        self.ffhq_data_path = ffhq_data_path
        self.celeba_data_path = celeba_data_path
        self.total_dataset = []
        
        self.same_prob = same_prob
        self.same_identity = same_identity
        
        ##data_path='/datasets/FFHQ'
        if bool(self.ffhq_data_path)==True:
            self.ffhq_dataset = glob.glob(f'{self.ffhq_data_path}/**/*.*g', recursive=True)
            self.ffhq_folders = glob.glob(f'{self.ffhq_data_path}/*', recursive=True)
            self.ffhq_len = len(self.ffhq_dataset)

        ##'/datasets/CelebHQ/CelebA-HQ-img'
        if bool(self.celeba_data_path)==True:
            self.celeba_dataset = glob.glob(f'{self.celeba_data_path}/**.*g', recursive=True)
            self.celeba_len = len(self.celeba_dataset)
        # ffhq_folders = glob.glob(f'{self.ffhq_data_path}/*', recursive=True)
        


        
        
        if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            self.total_dataset = self.ffhq_dataset + self.celeba_dataset 

        elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
            self.total_dataset += self.celeba_dataset 
        
        elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
            self.total_dataset += self.ffhq_dataset 
            
        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')
        

        self.folder2imgs = {}
        
        if bool(self.vgg_data_path)==True:
                    
            self.vgg_dataset = glob.glob(f'{self.vgg_data_path}/*/*.*g')
            self.vgg_folders_list = glob.glob(f'{self.vgg_data_path}/*')
            


            for folder in tqdm.tqdm(self.vgg_folders_list):
                folder_imgs = glob.glob(f'{folder}/*')
                self.folder2imgs[folder] = folder_imgs
          
            self.vgg_len = len(self.vgg_dataset)
            self.total_dataset += self.vgg_dataset
               
        if bool(self.dob_data_path)==True:
            '''
            디오비 데이터를 모두 로딩하기 위해 걸린시간: 1930.1818947792053초
            '''
            self.dob_dataset = glob.glob(f'{self.dob_data_path}/*/*/*.*g') ##모든이미지의 full image list 
            self.dob_folders_list = glob.glob(f'{self.dob_data_path}/*/*')  ##  '/datasets/DOB/imagefile/DOB04F215SEC', '/datasets/DOB/imagefile/KTT03M008FIR', ...
            # self.dob_folder2imgs = {}

            for folder in tqdm.tqdm(self.dob_folders_list): ##folder는 이미지를 가지고 있는 각각 폴더명
                folder_imgs = glob.glob(f'{folder}/*')  ##폴더명 아래에 있는 모든 파일을 가지고 와서 리스팅
                self.folder2imgs[folder] = folder_imgs ## 모든 이미지를 폴더경로 key에 저장하는 dict 생성. key의 예: folder2imgs['/datasets/DOB/imagefile2/CDW04F003FIR']
                
            self.dob_len = len(self.dob_dataset)
            self.total_dataset += self.dob_dataset
    
        self.total_dataset_len = len(self.total_dataset)
        
        
        
        self.transforms_id_ext = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            # transforms.RandomHorizontalFlip(p=0.4),  ##Hojun added
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):

        if bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
            # if idx < self.ffhq_len:
            #     raise ValueError('it must use only FFHQ dataset') 
            image_path = self.total_dataset[idx]
            transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xs, transforms_base_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person = self.fetch_ffhq_celeba(image_path)

        elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            if idx < self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xs, transforms_base_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person = self.fetch_ffhq_celeba(image_path)
                
            elif idx >= self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xs, transforms_base_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person =self.fetch_vgg_dob(image_path)
                
        # elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
        #     if idx < self.celeba_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_ffhq_celeba(image_path)
        #     elif idx >= self.celeba_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_vgg_dob(image_path)
                
        # elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
        #     if idx < self.ffhq_len:
        #         image_path = self.total_dataset[idx]
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_ffhq_celeba(image_path)
        #     elif idx >= self.ffhq_len:
        #         transforms_arcface_Xs, transforms_base_Xs,  transforms_base_Xt, same_person = self.fetch_vgg_dob(image_path)
        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')        

        return transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xs,  transforms_base_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person

    def __len__(self):
        return self.total_dataset_len

    def fetch_ffhq_celeba(self, image_path):
        # print('image_path', image_path)
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        ## 4가지 input (parsing) :데이터 구조 (원본 이미지 디렉토리와 같은 계층에 'parsed_img' 폴더가 있어야함)
        # img_folder_dir = '/'.join(image_path.split('/')[:-1])
        # Xsf = cv2.imread(img_parsed_dir + img_file_name + '_f.jpg')[:, :, ::-1]
        
        # Xsb = cv2.imread(img_parsed_dir + img_file_name + '_b.jpg')[:, :, ::-1]
        img_file_name = image_path.split('/')[-1].split('.')[-2]
        img_file_dir = image_path.split('/')[-2] + '/' + img_file_name
        img_parsed_dir = '/datasets/FFHQ_parsed_img/' + img_file_dir

        Xsf = cv2.imread(img_parsed_dir + '_f.jpg')[:, :, ::-1]
        Xsf = Image.fromarray(Xsf)
        Xsb = cv2.imread(img_parsed_dir + '_b.jpg')[:, :, ::-1]
        Xsb = Image.fromarray(Xsb)
        # img_type = image_path.split('/')[-1].split('.')[-1]
        # img_parsed_dir = '/parsed_img/'
        
        
        ######gdfgsdfgdfgsdgf
        #Xsf = cv2.imread(img_file_name + '_f.jpg')[:, :, ::-1]
        #Xsb = cv2.imread(img_folder_dir +  img_parsed_dir + img_file_name + '_b.jpg')[:, :, ::-1]

        
        # Xs = cv2.imread(image_path.split('/')[:-1] + '/FB')[:, :, ::-1]
        # Xs = Image.fromarray(Xs)

 

        if random.random() < self.same_prob:  ##모든 데이터를 다 reconstruction loss를 계산하지 않고 어느정도의 확률로만 계산하기 위한 것
            # image_path = random.choice(self.total_dataset[random.randint(0, len(self.total_dataset)-1)])  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            image_path = self.total_dataset[random.randint(0, len(self.total_dataset)-1)]  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)

            ## 4가지 input (parsing) :데이터 구조 (원본 이미지 디렉토리와 같은 계층에 'parsed_img' 폴더가 있어야함)
            img_file_name = image_path.split('/')[-1].split('.')[-2]
            img_file_dir = image_path.split('/')[-2] + '/' + img_file_name
            img_parsed_dir = '/datasets/FFHQ_parsed_img/' + img_file_dir
            Xtf = cv2.imread(img_parsed_dir + '_f.jpg')[:, :, ::-1]
            Xtb = cv2.imread(img_parsed_dir + '_b.jpg')[:, :, ::-1]
            Xtf = Image.fromarray(Xtf)
            Xtb = Image.fromarray(Xtb)
            
                    ######gdfgsdfgdfgsdgf
            #Xtf = cv2.imread(img_folder_dir +  img_parsed_dir + img_file_name + '_f.jpg')[:, :, ::-1]
            #Xtb = cv2.imread(img_folder_dir +  img_parsed_dir + img_file_name + '_b.jpg')[:, :, ::-1]
            
            same_person = 0
        else:
            Xt = Xs.copy()  ##확률적으로 가끔은 같은 Xs와 같은 이미지로 Xt를 사용해서 reconstruction loss 를 계산하기 위함 
            Xtf = Xsf.copy()
            Xtb = Xsb.copy()
            same_person = 1
            
        return self.transforms_id_ext(Xs),self.transforms_id_ext(Xt), self.transforms_base(Xs), self.transforms_base(Xt), self.transforms_base(Xtf), self.transforms_base(Xtb), self.transforms_base(Xsf), self.transforms_base(Xsb), same_person

    def fetch_vgg_dob(self, image_path):

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])

        if random.random() > self.same_prob:
            if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len+self.celeba_len-1, self.total_dataset_len-1)]

            elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
                image_path = self.total_dataset[random.randint(self.celeba_len-1, self.total_dataset_len-1)]
            
            elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len-1, self.total_dataset_len-1)]
                
            else:
                raise ValueError('At least either CelebA and FFHQ data must be used')
            
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person
    
    
    
class FaceEmbedCombined(TensorDataset):
    '''
    CelebA나 FFHQ중 하나는 무조건 사용되어야 하게 설정되어있다.
    '''
    def __init__(self, vgg_data_path=None, celeba_data_path=None, dob_data_path=None, ffhq_data_path=None, same_prob=0.8, same_identity=False):
        self.vgg_data_path = vgg_data_path
        self.dob_data_path = dob_data_path
        self.ffhq_data_path = ffhq_data_path
        self.celeba_data_path = celeba_data_path
        self.total_dataset = []
        self.same_prob = same_prob
        self.same_identity = same_identity
        ##data_path='/datasets/FFHQ'
        if bool(self.ffhq_data_path)==True:
            self.ffhq_dataset = glob.glob(f'{self.ffhq_data_path}/**/*f.jpg', recursive=True)
            self.ffhq_folders = glob.glob(f'{self.ffhq_data_path}/*', recursive=True)
            self.ffhq_len = len(self.ffhq_dataset)
        ##'/datasets/CelebHQ/CelebA-HQ-img'
        if bool(self.celeba_data_path)==True:
            self.celeba_dataset = glob.glob(f'{self.celeba_data_path}/**.*g', recursive=True)
            self.celeba_len = len(self.celeba_dataset)
        if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            self.total_dataset = self.ffhq_dataset + self.celeba_dataset
        elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
            self.total_dataset += self.celeba_dataset
        elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
            self.total_dataset += self.ffhq_dataset
        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')

        self.folder2imgs = {}
        if bool(self.vgg_data_path)==True:
            self.vgg_dataset = glob.glob(f'{self.vgg_data_path}/*/*.*g')
            self.vgg_folders_list = glob.glob(f'{self.vgg_data_path}/*')
            for folder in tqdm.tqdm(self.vgg_folders_list):
                folder_imgs = glob.glob(f'{folder}/*')
                self.folder2imgs[folder] = folder_imgs
            self.vgg_len = len(self.vgg_dataset)
            self.total_dataset += self.vgg_dataset
        if bool(self.dob_data_path)==True:
            '''
            디오비 데이터를 모두 로딩하기 위해 걸린시간: 1930.1818947792053초
            '''
            self.dob_dataset = glob.glob(f'{self.dob_data_path}/*/*/*.*g') ##모든이미지의 full image list
            self.dob_folders_list = glob.glob(f'{self.dob_data_path}/*/*')  ##  '/datasets/DOB/imagefile/DOB04F215SEC', '/datasets/DOB/imagefile/KTT03M008FIR', ...
            # self.dob_folder2imgs = {}
            for folder in tqdm.tqdm(self.dob_folders_list): ##folder는 이미지를 가지고 있는 각각 폴더명
                folder_imgs = glob.glob(f'{folder}/*')  ##폴더명 아래에 있는 모든 파일을 가지고 와서 리스팅
                self.folder2imgs[folder] = folder_imgs ## 모든 이미지를 폴더경로 key에 저장하는 dict 생성. key의 예: folder2imgs['/datasets/DOB/imagefile2/CDW04F003FIR']
            self.dob_len = len(self.dob_dataset)
            self.total_dataset += self.dob_dataset
        self.total_dataset_len = len(self.total_dataset)
        self.transforms_id_ext = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            # transforms.RandomHorizontalFlip(p=0.4),  ##Hojun added
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __getitem__(self, idx):
        if bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
            # if idx < self.ffhq_len:
            #     raise ValueError('it must use only FFHQ dataset')
            image_path = self.total_dataset[idx]
            transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person = self.fetch_ffhq_celeba(image_path)
        elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
            if idx < self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person = self.fetch_ffhq_celeba(image_path)
            elif idx >= self.ffhq_len + self.celeba_len:
                image_path = self.total_dataset[idx]
                transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person =self.fetch_vgg_dob(image_path)

        else:
            raise ValueError('At least either CelebA and/or FFHQ data must be used')
        return transforms_id_ext_Xs, transforms_id_ext_Xt, transforms_base_Xtf, transforms_base_Xtb, transforms_base_Xsf, transforms_base_Xsb, same_person

    def __len__(self):
        return self.total_dataset_len
    def fetch_ffhq_celeba(self, image_path):
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        img_file_name = image_path.split('/')[-1].split('_')[0]
        img_file_dir = image_path.split('/')[-2] + '/' + img_file_name
        img_parsed_dir = '/datasets/FFHQ_parsed_img/' + img_file_dir
        Xsf = cv2.imread(img_parsed_dir + '_f.jpg')[:, :, ::-1]
        Xsf = Image.fromarray(Xsf)
        Xsb = cv2.imread(img_parsed_dir + '_b.jpg')[:, :, ::-1]
        Xsb = Image.fromarray(Xsb)
        random_prob = random.random()
        if random_prob < self.same_prob:  ##모든 데이터를 다 reconstruction loss를 계산하지 않고 어느정도의 확률로만 계산하기 위한 것
            
            image_path = self.total_dataset[random.randint(0, len(self.total_dataset)-1)]  ##Xs에 대조될 Xt를 랜덤하게 뽑는것
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            ## 4가지 input (parsing) :데이터 구조 (원본 이미지 디렉토리와 같은 계층에 'parsed_img' 폴더가 있어야함)
            img_file_name = image_path.split('/')[-1].split('_')[0]
            img_file_dir = image_path.split('/')[-2] + '/' + img_file_name
            img_parsed_dir = '/datasets/FFHQ_parsed_img/' + img_file_dir
            Xtf = cv2.imread(img_parsed_dir + '_f.jpg')[:, :, ::-1]
            Xtb = cv2.imread(img_parsed_dir + '_b.jpg')[:, :, ::-1]
            Xtf = Image.fromarray(Xtf)
            Xtb = Image.fromarray(Xtb)
            same_person = 0
        else:
            Xt = Xs.copy()  ##확률적으로 가끔은 같은 Xs와 같은 이미지로 Xt를 사용해서 reconstruction loss 를 계산하기 위함
            Xtf = Xsf.copy()
            Xtb = Xsb.copy()
            same_person = 1
        return self.transforms_id_ext(Xs),self.transforms_id_ext(Xt), self.transforms_base(Xtf), self.transforms_base(Xtb), self.transforms_base(Xsf), self.transforms_base(Xsb), same_person
    def fetch_vgg_dob(self, image_path):
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])
        if random.random() > self.same_prob:
            if bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len+self.celeba_len-1, self.total_dataset_len-1)]
            elif bool(self.celeba_data_path)==True and bool(self.ffhq_data_path)==False:
                image_path = self.total_dataset[random.randint(self.celeba_len-1, self.total_dataset_len-1)]
            elif bool(self.celeba_data_path)==False and bool(self.ffhq_data_path)==True:
                image_path = self.total_dataset[random.randint(self.ffhq_len-1, self.total_dataset_len-1)]
            else:
                raise ValueError('At least either CelebA and FFHQ data must be used')
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person