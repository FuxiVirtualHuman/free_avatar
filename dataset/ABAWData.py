import shutil
from skimage import io
import os.path
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
# import face_alignment
from PIL import Image
import glob
import cv2
from torch.utils.data import DataLoader, Dataset
import csv
import sys
sys.path.append("..")
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))
from utils.tools import load_rigs_to_cache


def statistic(data):
    std = np.std(data)
    mean = np.mean(data)
    return


class ABAWDataset2(Dataset):
    def __init__(self, root_path, character,data_split, only_render=False, transform=None, random_flip=False, do_norm=True, 
                 use_ldmk=False, return_rigs=False, n_rigs=-1,faceware_ratio=0.1, img_postfix='.jpg'):
        self.character = character
        self.return_rigs = return_rigs
        self.n_rigs = n_rigs
        expr_fuxi_root = '/project/qiuf/Expr_fuxi/images_ttg_2024'
        self.faceware_root = os.path.join(root_path.replace('/images', '/faceware'))
        self.faceware_root_230 = '/project/qiuf/DJ01/L36_230/faceware'
        ttg_dy_root = '/project/qiuf/Expr_fuxi/images_ttg'
        fuxi_audio_data ='/project/qiuf/L36_drivendata/anrudong_2023_08_08_29_23_10_03_05_speaking_30fps/crop_face_processed'
        exclude_folder = [ 'L36Randsample', 'bichiyin_zuhe', 'L36face234_ZHY_PM_c101_230097_xia_51566_1',
                          'L36face234_ZHY_PM_c102_234011_banxiangzi_51560_1'] # 'qiufeng011'

        if character.lower() in ['l36_233', 'l36_234','l36_230', 'l36_230_61']:
            real_data = [
            '/data/Workspace/Rig2Face/data/yinxiaonv3',
            '/data/Workspace/Rig2Face/data/jiayang01',
            '/data/Workspace/Rig2Face/data/dongbu_kuazhang',
            '/data/Workspace/Rig2Face/data/singing',
            '/data/Workspace/Rig2Face/data/blackbro',
            '/data/Workspace/Rig2Face/data/Donbu_yinsu_31115_3_zhy',
            '/data/Workspace/Rig2Face/data/wjj01',
            '/data/Workspace/Rig2Face/data/qiufeng02',
            '/data/Workspace/Rig2Face/data/linjie_expr_test',
            '/project/qiuf/Expr_fuxi/images_old_50fps/C0016',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bxy02_52252_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_wjl_52251_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_mm_52247_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_lss_52250_1',
            '/project/qiuf/DJ01/L36/faceware/qiufeng011',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/wjj01',
            '/project/qiuf/DJ01/L36/faceware/wjj01',
            '/project/qiuf/DJ01/L36/faceware/qiufeng02',
            '/project/qiuf/DJ01/L36/faceware/qiufeng02',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            ]
            if character.lower() in ['l36_230', 'l36_230_61']:
                real_data += [
                            #   '/project/qiuf/DJ01/L36_230/faceware/230_first',
                            #   '/project/qiuf/DJ01/L36_230/faceware/230_second'
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                              ]
                              
            elif character.lower() in ['l36_233', 'l36_234']:
                real_data += [self.faceware_root]
                
            # 233faceware
            # faceware_folder = []
            if faceware_ratio:
                faceware_folder = os.listdir(self.faceware_root)[::int(1//faceware_ratio)]
                
                # faceware_folder += [y for x in os.walk(self.faceware_root) for y in glob.glob(os.path.join(x[0], '*.jpg'))][::int(1//faceware_ratio)]

                real_data += [os.path.join(self.faceware_root, e) for e in faceware_folder]
            # 230数据
            # self.faceware_folder_230 = self.load_faceware_230()
            # real_data += list(self.faceware_folder_230.values())

            # JHM大脸男的数据加一些
            # JHM_root = '/project/qiuf/DJ01/L36_230/faceware/230_second'
            # JHM_folder = [os.path.join(JHM_root, e) for e in os.listdir(JHM_root) if '_JHM_' in e and not e.endswith('.mp4')][::3]
            # real_data += JHM_folder

            # 新的女生数据加一些
            # GIRL_root = '/project/qiuf/Expr_fuxi/luoshen_data_clip/crop_face'
            # GIRL_folder = [os.path.join(GIRL_root, e) for e in os.listdir(GIRL_root) if 'player1_' in e][::10]
            # real_data += GIRL_folder

            # fuxi expr data
            # expr_fuxi_data = os.listdir(expr_fuxi_root)
            # real_data += [os.path.join(expr_fuxi_root, fo) for fo in expr_fuxi_data]

            # 伏羲audio data
            # read_data += fuxi_audio_data
            expr_audio_data = os.listdir(fuxi_audio_data)[::3]
            real_data += [os.path.join(fuxi_audio_data, fo) for fo in expr_audio_data] 

            # cartoon 
            # cartoon_root = '/project/qiuf/expr-capture/data2/crop_face'
            # cartoon_data = os.listdir(cartoon_root)[::]
            # real_data += [os.path.join(cartoon_root, fo) for fo in cartoon_data] *100
            
        self.root_path = root_path
        self.transform = transform
        self.use_ldmk = use_ldmk
        imgs_list = []
        self.rigs_list = []
        render_folders, render_folders_230 = [], []
        n_rig_fold = len(os.listdir(root_path.replace('/images', '/rigs')))
        image_list_action = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions{n_rig_fold}.txt')
        image_list_action_230 = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions_230.txt')
        if os.path.exists(image_list_action):
            with open(image_list_action, 'r') as f:
                lines = f.readlines()
            # lines = list(set(lines))
            render_folders += list(set([l.split('/')[0].split('_202')[0] for l in lines]))
            imgs_list += [os.path.join(root_path, l.strip()) for l in lines][::5]
        # if os.path.exists(image_list_action_230):
        #     with open(image_list_action_230, 'r') as f:
        #         lines_230 = f.readlines()
        #     render_folders_230 += list(set([l.split('/')[0] for l in lines_230]))
        #     imgs_list += [os.path.join(root_path.replace('/images', '/images_retarget_from_230'), l.strip()) for l in lines_230]
        if not imgs_list:
            # 如果没有另外指定list action
            folders = os.listdir(root_path)
            for folder in folders:
                if os.path.isdir(os.path.join(root_path, folder)):
                    images = os.listdir(os.path.join(root_path, folder))
                    imgs_list += [os.path.join(root_path, folder, imgname) for imgname in images]
        
        # load real data
        if not only_render:
            for folder in real_data:
                images = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                images += [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.png'))]
                if self.faceware_root_230 in folder:
                    images = images[::10]
                elif expr_fuxi_root in folder:
                    images = images[::10]
                if data_split in ['train', 'training']:
                    imgs_list += images[:int(0.9*len(images))]
                elif data_split in ['test']:
                    imgs_list += images[int(0.9*len(images)):]
        
        # 建立real data和render data之间的联系
        self.real2render = self._get_render_data_from_real_data(imgs_list, character)
        # self.real2render = {}
        
        self.imgs_list = imgs_list
        if not only_render:
            abaw_data = self.read_abaw_data(data_split)
            random.shuffle(abaw_data)
            abaw_data = abaw_data[:int(len(self.imgs_list)*0.3)]
            self.imgs_list += abaw_data[::]


        
        # self.imgs_list += self.read_MEAD_DATA(data_split, '/project/qiuf/MEAD/videos_res_no_smooth_selected_0511.txt')
        # self.imgs_list += self.read_MEAD_DATA(data_split, )
        # cache image data
        # self.img2rec = build_img2rec(root_path)
        # if return_rigs:
        #     self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # rigs_array =  np.array(list(self.rigs.values()))
        self.imgs_list = [e for e in self.imgs_list if e.split('/')[-2] not in exclude_folder]
        
        symm = [e for e in self.imgs_list if 'linjie' not in e and 'qiufeng' not in e and 'wjj' not in e ]
        dissymm = [e for e in self.imgs_list if 'linjie' in e or 'qiufeng' in e or 'wjj' in e ]
        self.imgs_list = symm[::2] + dissymm 
        # self.imgs_list = [e for e in self.imgs_list if 'linjie' in e and 'faceware' in e]
        # self.imgs_list = [e for e in self.imgs_list if 'linjie' in e]
        
        # 去除部分zhoumei数据
        self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # imgs_list_zhoumei = [l for l in self.imgs_list if f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}" in self.rigs and self.rigs[f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}"][14]>0.3]
        imgs_list_zhoumei = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][14]>0.4]
        imgs_list_minzui = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][38]>0.4]
        imgs_list_frownmouth = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][36]>0.4]
        imgs_list_blink_left = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             (self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][0]-self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][1]) < -0.1]
        
        imgs_list_press = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             (self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][0]-self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][38]) > 0.8]

        imgs_list_remove_zhoumei = [l for l in self.imgs_list if l not in imgs_list_zhoumei and l not in imgs_list_press]

        imgs_list_mouth_pucker = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][40]>0.8]

        self.imgs_list = imgs_list_remove_zhoumei+imgs_list_zhoumei[::3] + imgs_list_minzui*3+imgs_list_minzui*10+imgs_list_frownmouth*10 + imgs_list_blink_left*2+imgs_list_press+ imgs_list_mouth_pucker*10
        
        if character == 'L36_230_61':
            imgs_list_mouthright = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
                                (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) < -0.3 ]
            imgs_list_mouthleft = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
                                (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) > 0.3 ]
            imgs_list_mouth_pucker = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][40]>0.8]                    
            self.imgs_list = self.imgs_list+ imgs_list_mouthright*5+imgs_list_mouthleft*5


# [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
#                              (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) > 0.3 ]

        # self.bs_detector = FaceDetectorMediapipe()
        # self.bs = self._get_bs_cache(character, data_split)
        self.bs = {}
        # self.imgs_list = self.imgs_list[:500]
        # self.imgs_list = [e for e in self.imgs_list if 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1' in e]
        print(f'{data_split} data: {len(self.imgs_list)}')
    
    def _get_bs_cache(self, character, data_split):
        bs_cache_pkl = f'/project/qiuf/DJ01/{character}/bs_20231228_{data_split}.pkl'
        bs_cache = {}

        if os.path.exists(bs_cache_pkl):
            with open(bs_cache_pkl, 'rb') as f:
                bs_cache = pickle.load(f)

        imgs_list_rest = [im for im in self.imgs_list if im not in bs_cache]
        if not imgs_list_rest:
            return bs_cache
        
        for img_path in tqdm(imgs_list_rest):
            bs, _, has_face = self.bs_detector.detect_from_PIL(Image.open(img_path).convert('RGB'))
            if has_face:
                bs_cache[img_path] = np.array(bs)
            else:
                bs_cache[img_path] = np.zeros(52)    
        with open(bs_cache_pkl, 'wb') as f:
            pickle.dump(bs_cache, f)
        return bs_cache

    def _get_render_data_from_real_data(self, imgs_list, character):
        real2render = {}
        if character in ['L36_233', 'L36_230', 'L36_230_61']:
            character = character.replace('_233', '')
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231229.pkl'

            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                elif self.faceware_root_230 in img:
                    render_root = f'/project/qiuf/DJ01/{character}/images_retarget_from_230'
                else:
                    continue
                # if 'linjie_expr_test' in img:
                #     print(img)
                vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                img_index = int(imgname.split('.')[0].split('_')[0])
                imgname_render = f'{img_index:06d}.0000.jpg'

                imgpath = os.path.join(os.path.dirname(img).replace('faceware/', 'images/'), imgname_render)
                _imgpath = imgpath.replace('.jpg', '.png')
                if os.path.exists(imgpath):
                    real2render[img] = imgpath
                elif os.path.exists(_imgpath):
                    real2render[img] = _imgpath
                    
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        elif character=='L36_234':
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231122.pkl'
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                    vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                    img_index = imgname.split('.')[0]
                    imgname_render = f'{img_index}.0000.png'
                    imgpath = os.path.join(render_root,vid_folder, imgname_render)
                    if os.path.exists(imgpath):
                        real2render[img] = imgpath
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        return real2render
    
    def load_faceware_230(self):
        folders = ['/project/qiuf/DJ01/L36_230/faceware/230_first', 
                   '/project/qiuf/DJ01/L36_230/faceware/230_second']
        faceware_230 = {}
        for fold in folders:
            videos = os.listdir(fold)
            for v in videos:
                faceware_230[v] = os.path.join(fold, v)
        return faceware_230
    
    def read_MEAD_DATA(self, data_split, list_path_selected=''):
        root = '/project/qiuf/MEAD/images'
        folders = os.listdir(root)[::5]
        if data_split in ['train' or 'training']:
            folders = folders[:int(len(folders)*0.9)]
        else:
            folders = folders[int(len(folders)*0.9):]
        img_list=[]
        for folder in folders:
            imgnames = os.listdir(os.path.join(root, folder))
            img_list += [os.path.join(root, folder, im) for im in imgnames][::10]
        # 效果不错的视频，做强制对齐
        self.MEAD_selected={}
        if list_path_selected:
            with open(list_path_selected, 'r') as f:
                MEAD_selected = f.readlines()
            for line in MEAD_selected:
                self.MEAD_selected[line.split('_2022')[0]] = line.strip()
            # self.MEAD_selected = [ms.strip() for ms in MEAD_selected]
        return img_list

    def read_abaw_data(self, data_split):
        root = '/data/data/ABAW/crop_face_jpg'
        path_pkl = f'/data/Workspace/Rig2Face/data/abaw_images_{data_split}_large.pkl'
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                img_list = pickle.load(f)
            return img_list[:40000]

        if data_split == 'train':
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_training.csv'
        else:
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_validation.csv'
        with open(data_file, newline='') as f:
            data_list = list(csv.reader(f, delimiter=' '))
        img_list = [os.path.join(root, d[0].split(',')[1]) for d in data_list[1:]]
        img_list = [ad for ad in img_list if os.path.exists(ad)][:50000]
        with open(path_pkl, 'wb') as f:
            pickle.dump(img_list, f)
        return img_list

    def __getitem__(self, index):
        img_path = self.imgs_list[index]

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f'reading img error:{img_path}')
            img_path = self.imgs_list[20]
            img = Image.open(img_path).convert('RGB')

        # todo:
        if self.root_path in img_path or 'render' in img_path :
            is_render = 1
            do_pixel = 1
        else:
            is_render = 0
            do_pixel = 0
        
        if img_path in self.bs:
            bs = self.bs[img_path]
        else:
            # print('bs not exist')
            bs = np.zeros(52)    
        if 'Rig2Face/data' in img_path:
            role_id = 1
        elif 'aligned' in img_path:
            role_id = 0
        else:
            role_id = 2   # render data
        
            
        # l36动捕数据强制和动画数据对齐。        
        if img_path in self.real2render:
            do_pixel = 1 # 1才会做像素loss
            target_path = self.real2render[img_path]
            try:
                target = Image.open(target_path).convert('RGB')
            except:
                target = img
        else:
            target = img
            
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        data = {'img':img, 'target':target, 'is_render':is_render, 'ldmk':np.array([0]*136), 'role_id':role_id, 'do_pixel':do_pixel, 'bs':bs}
        if self.return_rigs:
            _img_path_split = img_path.split('/')[6:]
            imgname = _img_path_split[-1]
            try:
                vindex = int(imgname.split('.')[0])
                rigname = f'{vindex:06d}.txt'
            except:
                rigname = imgname.replace('.jpg', '.txt')
            rigname = '/'.join(_img_path_split[:-1]+[rigname])
            
            # foldname, imgname = img_path.split('/')[-2:]
            # try:
            #     vindex = int(imgname.split('.')[0])
            #     rigname = f'{vindex:06d}.txt'
                
            # except:
            #     rigname = imgname.replace('.jpg', '.txt')
            # rigname = f"{foldname}/{rigname}"
            if rigname in self.rigs:
                rigs = np.array(self.rigs[rigname])
                has_rig = 1
                # print(rigname)
            else:
                assert do_pixel == 0, f'do pixel loss but no rigs:{rigname}'
                rigs = np.zeros(self.n_rigs)
                has_rig = 0
            data['rigs'] = rigs
        data['has_rig'] = has_rig
        return data
    
    def __len__(self):
        return len(self.imgs_list)


class ABAWDataset2_multichar(Dataset):
    def __init__(self, characters, data_split, CHARACTER_NAME, transform=None, random_flip=False, do_norm=True, use_ldmk=False, return_rigs=False):
        self.return_rigs = return_rigs
        self.imgs_list = []
        self.real2render = {}
        self.rigs = {}
        self.render_path = []
        self.n_rigs = 0
        self.CHARACTER_NAME = CHARACTER_NAME
        for character, character_config in characters.items():    
            root_path, n_rigs= character_config['data_path'], character_config['n_rig']
            self.render_path.append(root_path+'/')
            self.character = character
            self.n_rigs = max(self.n_rigs, n_rigs)
            expr_fuxi_root = '/project/qiuf/Expr_fuxi/images_ttg_2024'
            self.faceware_root = os.path.join(root_path.replace('/images', '/faceware'))
            self.faceware_root_230 = '/project/qiuf/DJ01/L36_230/faceware'
            ttg_dy_root = '/project/qiuf/Expr_fuxi/images_ttg'
            fuxi_audio_data ='/project/qiuf/L36_drivendata/anrudong_2023_08_08_29_23_10_03_05_speaking_30fps/crop_face_processed'
            exclude_folder = [ 'L36Randsample', 'bichiyin_zuhe', 'L36face234_ZHY_PM_c101_230097_xia_51566_1',
                            'L36face234_ZHY_PM_c102_234011_banxiangzi_51560_1'] # 'qiufeng011'

            if character.lower() in ['l36_233', 'l36_234','l36_230', 'l36_230_61']:
                real_data = [
                '/data/Workspace/Rig2Face/data/yinxiaonv3',
                '/data/Workspace/Rig2Face/data/jiayang01',
                '/data/Workspace/Rig2Face/data/dongbu_kuazhang',
                '/data/Workspace/Rig2Face/data/singing',
                '/data/Workspace/Rig2Face/data/blackbro',
                '/data/Workspace/Rig2Face/data/Donbu_yinsu_31115_3_zhy',
                '/data/Workspace/Rig2Face/data/wjj01',
                '/data/Workspace/Rig2Face/data/qiufeng02',
                '/data/Workspace/Rig2Face/data/linjie_expr_test',
                '/project/qiuf/Expr_fuxi/images_old_50fps/C0016',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bxy02_52252_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_wjl_52251_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_mm_52247_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_lss_52250_1',
                '/project/qiuf/DJ01/L36/faceware/qiufeng011',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/wjj01',
                '/project/qiuf/DJ01/L36/faceware/wjj01',
                '/project/qiuf/DJ01/L36/faceware/qiufeng02',
                '/project/qiuf/DJ01/L36/faceware/qiufeng02',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                ]
                if character.lower() == 'l36_230':
                    real_data += ['/project/qiuf/DJ01/L36_230/faceware/230_first',
                                '/project/qiuf/DJ01/L36_230/faceware/230_second'
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                ]
                                
                elif character.lower() in ['l36_233', 'l36_234']:
                    real_data += [self.faceware_root]
                    
                # 233faceware
                faceware_folder = os.listdir(self.faceware_root)
                # faceware_folder = [y for x in os.walk(self.faceware_root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                real_data += [os.path.join(self.faceware_root, e) for e in faceware_folder]
                # 230数据
                # self.faceware_folder_230 = self.load_faceware_230()
                # real_data += list(self.faceware_folder_230.values())

                # JHM大脸男的数据加一些
                # JHM_root = '/project/qiuf/DJ01/L36_230/faceware/230_second'
                # JHM_folder = [os.path.join(JHM_root, e) for e in os.listdir(JHM_root) if '_JHM_' in e and not e.endswith('.mp4')][::3]
                # real_data += JHM_folder

                # 伏羲audio data
                # read_data += fuxi_audio_data
                expr_audio_data = os.listdir(fuxi_audio_data)[::3]
                real_data += [os.path.join(fuxi_audio_data, fo) for fo in expr_audio_data] 

                
            self.root_path = root_path
            self.transform = transform
            self.use_ldmk = use_ldmk
            imgs_list = []
            render_folders, render_folders_230 = [], []
            n_rig_fold = len(os.listdir(root_path.replace('/images', '/rigs')))
            image_list_action = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions{n_rig_fold}.txt')
            if os.path.exists(image_list_action):
                with open(image_list_action, 'r') as f:
                    lines = f.readlines()
                # lines = list(set(lines))
                render_folders += list(set([l.split('/')[0].split('_202')[0] for l in lines]))
                imgs_list += [os.path.join(root_path, l.strip()) for l in lines]
                
            image_list_action_230 = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions_230.txt')
            if os.path.exists(image_list_action_230):
                with open(image_list_action_230, 'r') as f:
                    lines_230 = f.readlines()
                render_folders_230 += list(set([l.split('/')[0] for l in lines_230]))
                imgs_list += [os.path.join(root_path.replace('/images', '/images_retarget_from_230'), l.strip()) for l in lines_230]
            if not imgs_list:
                # 如果没有另外指定list action
                folders = os.listdir(root_path)
                for folder in folders:
                    if os.path.isdir(os.path.join(root_path, folder)):
                        images = os.listdir(os.path.join(root_path, folder))
                        imgs_list += [os.path.join(root_path, folder, imgname) for imgname in images]
            
            # load real data
            for folder in real_data:
                images = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                images += [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.png'))]
                if self.faceware_root_230 in folder:
                    images = images[::10]
                elif expr_fuxi_root in folder:
                    images = images[::10]
                if data_split in ['train', 'training']:
                    imgs_list += images[:int(0.9*len(images))]
                elif data_split in ['test']:
                    imgs_list += images[int(0.9*len(images)):]
            
            # 建立real data和render data之间的联系
            self.real2render[character]=self._get_render_data_from_real_data(imgs_list, character)
            # self.real2render = {}
            

            abaw_data = self.read_abaw_data(data_split)
            random.shuffle(abaw_data)
            abaw_data = abaw_data[:int(len(imgs_list)*0.3)]
            imgs_list += abaw_data[::]

            imgs_list = [e for e in imgs_list if e.split('/')[-2] not in exclude_folder]
            
            symm = [e for e in imgs_list if 'linjie' not in e and 'qiufeng' not in e and 'wjj' not in e ]
            dissymm = [e for e in imgs_list if 'linjie' in e or 'qiufeng' in e or 'wjj' in e ]
            imgs_list = symm[::2] + dissymm 
            # imgs_list = [e for e in imgs_list if 'linjie' in e and 'faceware' in e]
            # imgs_list = [e for e in imgs_list if 'linjie' in e]
            
            # 去除部分zhoumei数据
            self.rigs[character]=load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
            # imgs_list_zhoumei = [l for l in imgs_list if f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}" in self.rigs and self.rigs[f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}"][14]>0.3]
            imgs_list_zhoumei = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][14]>0.3]
            imgs_list_minzui = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][38]>0.6]
            imgs_list_frownmouth = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][36]>0.3]
            imgs_list_blink_left = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                (self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][0]-self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][1]) < -0.1]
            imgs_list_duzui = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][40]>0.6 and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][41]>0.6 ]
            imgs_list_guzui =  [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][64 if n_rigs==67 else 48]>0.6]
            imgs_list_press = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                (self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][0]-self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][38]) > 0.8]
            imgs_list_smile =  [l for l in imgs_list if character=='L36_230_61' and l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][54]>0.6 and self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][19]<0.1]
            imgs_list_remove_zhoumei = [l for l in imgs_list if l not in imgs_list_zhoumei and l not in imgs_list_press]
            imgs_list = imgs_list_remove_zhoumei+imgs_list_zhoumei[::5] + imgs_list_minzui*5 + imgs_list_frownmouth*10 + imgs_list_blink_left*2 + \
                imgs_list_press[::3] + imgs_list_duzui*10 + imgs_list_guzui * 5 +  imgs_list_smile *5
            self.imgs_list+=[[e, CHARACTER_NAME.index(character)] for e in imgs_list]
            # self.bs_detector = FaceDetectorMediapipe()
            # self.bs = self._get_bs_cache(character, data_split)
        self.bs = {}
            # self.imgs_list = self.imgs_list[:500]
            # self.imgs_list = [e for e in self.imgs_list if 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1' in e]
        # self.imgs_list = self.imgs_list[:80]
        print(f'{data_split} data: {len(self.imgs_list)}')
    
    def _get_bs_cache(self, character, data_split):
        bs_cache_pkl = f'/project/qiuf/DJ01/{character}/bs_20231228_{data_split}.pkl'
        bs_cache = {}

        if os.path.exists(bs_cache_pkl):
            with open(bs_cache_pkl, 'rb') as f:
                bs_cache = pickle.load(f)

        imgs_list_rest = [im for im in self.imgs_list if im not in bs_cache]
        if not imgs_list_rest:
            return bs_cache
        
        for img_path in tqdm(imgs_list_rest):
            bs, _, has_face = self.bs_detector.detect_from_PIL(Image.open(img_path).convert('RGB'))
            if has_face:
                bs_cache[img_path] = np.array(bs)
            else:
                bs_cache[img_path] = np.zeros(52)    
        with open(bs_cache_pkl, 'wb') as f:
            pickle.dump(bs_cache, f)
        return bs_cache

    def _get_render_data_from_real_data(self, imgs_list, character):
        real2render = {}
        if character in ['L36_233', 'L36_230', 'L36_230_61']:
            character = character.replace('_233', '')
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231229.pkl'

            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                elif self.faceware_root_230 in img:
                    render_root = f'/project/qiuf/DJ01/{character}/images_retarget_from_230'
                else:
                    continue
                # if 'linjie_expr_test' in img:
                #     print(img)
                vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                img_index = int(imgname.split('.')[0].split('_')[0])
                imgname_render = f'{img_index:06d}.0000.jpg'

                imgpath = os.path.join(os.path.dirname(img).replace('faceware/', 'images/'), imgname_render)
                if os.path.exists(imgpath):
                    real2render[img] = imgpath
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        elif character=='L36_234':
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231122.pkl'
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                    vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                    img_index = imgname.split('.')[0]
                    imgname_render = f'{img_index}.0000.png'
                    imgpath = os.path.join(render_root,vid_folder, imgname_render)
                    if os.path.exists(imgpath):
                        real2render[img] = imgpath
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        return real2render
    
    def load_faceware_230(self):
        folders = ['/project/qiuf/DJ01/L36_230/faceware/230_first', 
                   '/project/qiuf/DJ01/L36_230/faceware/230_second']
        faceware_230 = {}
        for fold in folders:
            videos = os.listdir(fold)
            for v in videos:
                faceware_230[v] = os.path.join(fold, v)
        return faceware_230
    
    def read_MEAD_DATA(self, data_split, list_path_selected=''):
        root = '/project/qiuf/MEAD/images'
        folders = os.listdir(root)[::5]
        if data_split in ['train' or 'training']:
            folders = folders[:int(len(folders)*0.9)]
        else:
            folders = folders[int(len(folders)*0.9):]
        img_list=[]
        for folder in folders:
            imgnames = os.listdir(os.path.join(root, folder))
            img_list += [os.path.join(root, folder, im) for im in imgnames][::10]
        # 效果不错的视频，做强制对齐
        self.MEAD_selected={}
        if list_path_selected:
            with open(list_path_selected, 'r') as f:
                MEAD_selected = f.readlines()
            for line in MEAD_selected:
                self.MEAD_selected[line.split('_2022')[0]] = line.strip()
            # self.MEAD_selected = [ms.strip() for ms in MEAD_selected]
        return img_list

    def read_abaw_data(self, data_split):
        root = '/data/data/ABAW/crop_face_jpg'
        path_pkl = f'/data/Workspace/Rig2Face/data/abaw_images_{data_split}_large.pkl'
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                img_list = pickle.load(f)
            return img_list[:40000]

        if data_split == 'train':
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_training.csv'
        else:
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_validation.csv'
        with open(data_file, newline='') as f:
            data_list = list(csv.reader(f, delimiter=' '))
        img_list = [os.path.join(root, d[0].split(',')[1]) for d in data_list[1:]]
        img_list = [ad for ad in img_list if os.path.exists(ad)][:50000]
        with open(path_pkl, 'wb') as f:
            pickle.dump(img_list, f)
        return img_list

    def __getitem__(self, index):
        img_path, role_id = self.imgs_list[index]
        character = self.CHARACTER_NAME[role_id]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f'reading img error:{img_path}')
            # _imgname = img_path.split('/').split('.')[0]
            # img_path = img_path.replace(_imgname, f'{int(_imgname)-1:06d}')
            img_path, role_id = self.imgs_list[128]
            img = Image.open(img_path).convert('RGB')

        # todo:
        _img_path_split = img_path.split('/')[6:]
        imgname = _img_path_split[-1]
        try:
            vindex = int(imgname.split('.')[0])
            rigname = f'{vindex:06d}.txt'
        except:
            rigname = imgname.replace('.jpg', '.txt')
        rigname = '/'.join(_img_path_split[:-1]+[rigname])
            
        is_render = 0
        do_pixel = 0
        if 'render' in img_path and rigname in self.rigs[character]:
            is_render = 1
            do_pixel = 1 
        else:
            for render_path in self.render_path:
                if render_path in img_path and rigname in self.rigs[character]:
                    is_render = 1
                    do_pixel = 1    
        
        if img_path in self.bs:
            bs = self.bs[img_path]
        else:
            bs = np.zeros(52)    

            
        # l36动捕数据强制和动画数据对齐。        
        if img_path in self.real2render[character]:
            do_pixel = 1 # 1才会做像素loss
            target_path = self.real2render[character][img_path]
            target = Image.open(target_path).convert('RGB')
            # target = img

        else:
            target = img
            
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        data = {'img':img, 'target':target, 'is_render':is_render, 'ldmk':np.array([0]*136), 'role_id':role_id, 'do_pixel':do_pixel, 'bs':bs}
        if self.return_rigs:
            rigs = np.zeros(self.n_rigs)
            # for render_path in self.render_path:
            #     img_path = img_path.replace(render_path, '')
            # _img_path_split = img_path.split('/')[6:]
            # imgname = _img_path_split[-1]
            # try:
            #     vindex = int(imgname.split('.')[0])
            #     rigname = f'{vindex:06d}.txt'
            # except:
            #     rigname = imgname.replace('.jpg', '.txt')
            # rigname = '/'.join(_img_path_split[:-1]+[rigname])
            if rigname in self.rigs[character]:
                
                _rigs = np.array(self.rigs[character][rigname])
                rigs[:len(_rigs)] = _rigs
                has_rig = 1
                # print(rigname)
            else:
                if do_pixel==1:
                    print(1)
                assert do_pixel == 0, f'do pixel loss but no {character} rigs:{rigname}'
                has_rig = 0
            data['rigs'] = rigs
            data['has_rig'] = has_rig
        return data
    
    def __len__(self):
        return len(self.imgs_list)
    
def center_crop_restore(img):
    # input: 256*256*3
    w, h = 211, 211

    if img.shape[0] == 3:
        ori_shape = img.shape

        x = int(img.shape[1] / 2 - w / 2)
        y = int(img.shape[2] / 2 - h / 2)
        crop_img = img[:, y:y+h,x:x+w]
        img = np.resize(crop_img, ori_shape)
        return img
    else:
        ori_shape = img.shape

        x = int(img.shape[0] / 2 - w / 2)
        y = int(img.shape[1] / 2 - h / 2)
        crop_img = img[y:y+h,x:x+w]
        img = cv2.resize(crop_img, (ori_shape[0], ori_shape[1]))
        return img

def read_image(img_path, mode='rgb', flip=False, size=None, center_crop=True, div=False):
    img = cv2.imread(img_path)
    try:
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print('read image error:', img_path)
    if flip:
        img = cv2.flip(img, 1)
    if size:
        img = cv2.resize(img, (size,size))
    if center_crop:
        img = center_crop_restore(img)
    if div:
        img = img / 255.
    return img

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [list(map(float, l.strip(' \n')[:-1].replace(',',' ').split(' '))) for l in lines[1:]]
    return lines

def extract_DJ01_images():
    from glob import glob
    # root = '/data/data/DJ01/DJ01/images'
    root = '/data/Workspace/Rig2Face/results/kanghui220220402-023345test/'
    images = [y for x in os.walk(root) for y in glob(os.path.join(x[0], '*.jpg'))]
    # images = images[:5000:3] + images[::10]
    images = images[::3]
    save_root = os.path.join(root, 'query_dj')
    os.makedirs(save_root, exist_ok=True)
    for im_source in tqdm(images):
        target = os.path.join(save_root, im_source.split('/')[-1])
        shutil.copyfile(im_source, target)
    return

def make_debert_batch():
    file_feature = os.path.join('/data/data/ABAW5/challenge4/', 'val_hubert.pkl')
    save_root = file_feature.replace('.pkl','')
    os.makedirs(save_root,exist_ok=True)
    with open(file_feature, 'rb') as f:
        features = pickle.load(f)
    for vid_id in tqdm(features.keys(),total=len(features)):
        save_path = os.path.join(save_root, vid_id)
        with open(save_path, 'wb') as f:
            pickle.dump(features[vid_id], f)         
            
def read_L36_ctrls(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [list(map(float,l.strip(' \n').split(' '))) for l in lines]
    return np.array(lines).squeeze()


def filter_image_data_with_actions_233():
    random.seed(0)
    root = r'/project/qiuf/DJ01/L36/images'
    save_path = '/project/qiuf/DJ01/L36/images_train_list_actions.txt'
    folders = os.listdir(root)
    data_list = []
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    random.shuffle(folders)
    folders = folders[:int(len(folders)*0.9)]
    
    avaible_folders = []
    # TODO: 统计不同维度的出现频次来重采样数据。只算嘴
    with open(r'/project/qiuf/DJ01/L36/rigs71.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    rigs = [] 
    
    for name, rig in rigs_data.items():
        if '233_' in name or 'CYY_' in name or 'mainline_' in name:
        # if '233_chikouxiongceshi_30082_1' in name:
            rigs.append(rig)
    rigs_avg = np.mean(np.array(rigs),axis=0)
    max_mouth = np.max(rigs_avg[20:])
    rigs_freq_normed = rigs_avg/max_mouth
    rigs_sample_freq = 1/rigs_freq_normed
    count_no_action = 0
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    rigs_mouth = np.array(rigs)[:, 20:][::10]
    for e, folder in enumerate(folders):
        if '233_' not in folder and 'CYY_' not in folder and 'mainline_' not in folder:
            continue
        print(f'[{e}/{len(folders)}]:', folder)
        imgnames = os.listdir(os.path.join(root, folder))
        try:
           imgnames.sort(key=lambda x:int(x.split('.')[0]))
        except:
            continue
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        avaible_folders.append(folder)
        
        sample_freqs = []
        for i, imgname in tqdm(enumerate(imgnames)):
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[20:], axis=1)
            cnt_similar = np.sum(dist<0.5) 
            sample_freq = np.log(coff_count/cnt_similar)
            # sample_freqs.append(sample_freq)
            if sample_freq < 1:
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
            

            # 给每一个rig维度都附一个采样频率，然后判断每个rig是否有动作，然后取最大值进行重复采样。
            # 但是这样独立地统计每一个rig的频率并不能很好地表征动作出现的频率。
            # rig_action_index = np.where(rig>0.2)[0]
            # rig_action_index_mouth = np.array([i for i in rig_action_index if i > 19])
            # if rig_action_index_mouth.shape[0]>0:
            #     sample_n = np.max(rigs_sample_freq[rig_action_index_mouth])
            # else:
            #     count_no_action += 1
            #     sample_n = 0.1 # 如果没动作 采样频率是0.1
            # if sample_n < 1:
            #     if np.random.randn() < sample_n:
            #         data_list.append(f'{folder}/{imgname}\n')
            # else:
            #     for _ in range(int(sample_n)):
            #         data_list.append(f'{folder}/{imgname}\n')

            # 根据rig的最大值，来判断是否有动作，有动作的进行采样，没动作的进行0.1频率的重采样
            # if np.max(rig[2:]) > 0.3:
            #     data_list.append(f'{folder}/{imgname}\n')
            #     # print('avaiable image', imgname)
            # elif np.random.randn() < 0.1:
            #     data_list.append(f'{folder}/{imgname}\n')
    sample_freqs = np.array(sample_freqs)
    sample_freqs_log = np.log(sample_freqs)
    print('samples with no actions', count_no_action)
    rigs = np.array(rigs)
    print('avaible folder: ', avaible_folders)
    print('total data:', len(data_list))
    with open(save_path, 'w') as f:
        f.writelines(data_list)
    
def filter_image_data_with_actions_233_clean():
    random.seed(0)
    root = r'/project/qiuf/DJ01/L36/images'
    save_path = '/project/qiuf/DJ01/L36/images_test_list_actions666.txt'
    folders = os.listdir(root)
    data_list = []
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    random.shuffle(folders)
    folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    # folders += ['qiufeng011', 'wjj01', 'qiufeng02', 'qiufeng03', 'wjj01_blink', 'qiufeng02_blink', 
                # 'qiufeng03_blink', 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1', 'L36face233_ZXS_PM_233_nielian_c08_55212_1']
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    
    avaible_folders = []
    # 统计不同维度的出现频次来重采样数据。只算嘴
    with open(r'/project/qiuf/DJ01/L36/rigs666.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    rigs = np.array(list(rigs_data.values()))[::10]
    # rigs = []
    
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    rigs_mouth = np.array(rigs)[:, 19:]
    for e, folder in enumerate(folders):
        print(f'[{e}/{len(folders)}]:', folder)
        imgnames = os.listdir(os.path.join(root, folder))
        try:
           imgnames.sort(key=lambda x:int(x.split('.')[0]))
        except:
            continue
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        avaible_folders.append(folder)
        
        for i, imgname in tqdm(enumerate(imgnames)):
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[19:], axis=1)
            cnt_similar = np.sum(dist<0.5) 
            sample_freq = np.log(coff_count/cnt_similar)
            sample_freq = min(max(0.01, sample_freq), 10) # 控制在10倍和0.1倍之间。 
            if sample_freq < 1:
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
            
    print('avaible folder: ', avaible_folders)
    print('total data:', len(data_list))
    with open(save_path, 'w') as f:
        f.writelines(data_list)
        
def filter_image_data_with_actions_230_clean():
    random.seed(0)
    
    root = r'/project/qiuf/DJ01/L36_230_61/images'
    rig_folders_n = '321'

    folders = os.listdir(root)
    data_list = []
    # folders = ['230_first', '230_second', 'L36_230#230_233_qujia_nanwanjia_60071_4', 'L36_230#230_233_huazhang_60069_2',
    #            'L36_230#233_fanghedeng02_59838_1', 'L36_230#230_233_wangtian_60064_4']
    random.shuffle(folders)
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    # folders += ['qiufeng011', 'wjj01', 'qiufeng02', 'qiufeng03', 'wjj01_blink', 'qiufeng02_blink', 
                # 'qiufeng03_blink', 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1', 'L36face233_ZXS_PM_233_nielian_c08_55212_1']
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    
    avaible_folders = []
    # 统计不同维度的出现频次来重采样数据。只算嘴
    with open(root.replace('images', 'rigs')+str(rig_folders_n)+'.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    # rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    rigs = np.array(list(rigs_data.values()))[::10]
    # rigs = []
    
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    rigs_mouth = np.array(rigs)[:, 19:]
    for e, folder in enumerate(folders):
        print(f'[{e}/{len(folders)}]:', folder)
        imgnames = [y for x in os.walk(os.path.join(root, folder)) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
        imgnames += [y for x in os.walk(os.path.join(root, folder)) for y in glob.glob(os.path.join(x[0], '*.png'))]
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        # rigs_files =  [y for x in os.walk(rigs_folder) for y in glob.glob(os.path.join(x[0], '*.txt'))]
        # if len(imgnames) != len(rigs_files):
        #     print(f'{folder} images and rigs has different frames!')
        #     continue
        avaible_folders+=(list(set([e.split('/')[-2] for e in imgnames])))
        
        for i, imgname in tqdm(enumerate(imgnames)):
            rigname = imgname.replace('.0000.jpg', '.txt').replace(root, '').strip('/')
            rigname = imgname.replace('.0000.png', '.txt').replace(root, '').strip('/')
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                try:
                    rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                    rig = read_L36_ctrls(os.path.join(root, rigname))
                except:
                    print(f'{rigname} not exists')            
                    continue
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[19:], axis=1)
            cnt_similar = np.sum(dist<0.5) 
            sample_freq = np.log(coff_count/cnt_similar)
            sample_freq = min(max(0.01, sample_freq), 10) # 控制在10倍和0.1倍之间。 
            if sample_freq < 1:
                if np.random.randn() < sample_freq:
                    data_list.append(imgname.replace(root, '').strip('/')+'\n')
            else:
                for _ in range(int(sample_freq)):
                    data_list.append(imgname.replace(root, '').strip('/')+'\n')
            

    test_folder = avaible_folders[int(len(avaible_folders)*0.9):]
    test_list = [e for e in data_list if e.split('/')[-2] in test_folder]
    save_path = f'{root}_test_list_actions{rig_folders_n}.txt'
    with open(save_path, 'w') as f:
        f.writelines(test_list)
        
    train_folder = avaible_folders[:int(len(avaible_folders)*0.9)]
    # train_folder += folders[2:]
    train_list = [e for e in data_list if e.split('/')[-2] in train_folder]
    save_path = f'{root}_train_list_actions{rig_folders_n}.txt'
    with open(save_path, 'w') as f:
        f.writelines(train_list)
    print('total data:', len(data_list))
    

    print('train data:', len(train_list))
    print('train folder: ', train_folder)

    print('test data:', len(test_list))
    print('test folder: ', test_folder)
    
def filter_image_data_with_actions_230():
    random.seed(0)
    root = r'/project/qiuf/DJ01/L36/images_retarget_from_230'
    save_path = '/project/qiuf/DJ01/L36/images_test_list_actions_230.txt'
    folders = os.listdir(root)
    data_list = []
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    random.shuffle(folders)
    folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    
    avaible_folders = []
    # 统计不同维度的出现频次来重采样数据。只算嘴
    rigs_root = root.replace('images', 'rigs')
    rigs_pkl = rigs_root + '.pkl'
    if os.path.exists(rigs_pkl):        
        with open(rigs_pkl, 'rb') as f:
            rigs_data = pickle.load(f)
        rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    else:
        rigs_data = {}
        rig_folders = os.listdir(rigs_root)
        for rig_folder in tqdm(rig_folders, total=len(rig_folders)):
            rig_files = os.listdir(os.path.join(rigs_root, rig_folder))
            for rigf in rig_files:
                rig_path = os.path.join(rigs_root, rig_folder, rigf)
                rig = read_L36_ctrls(rig_path)
                rigs_data[os.path.join(rig_folder, rigf)] = rig
        with open(rigs_pkl, 'wb') as f:
            pickle.dump(rigs_data, f)
    rigs = np.array(list(rigs_data.values()))
    rigs= np.array([np.array(rig) for rig in rigs if len(rig)==61])
    print(rigs.shape)
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    rigs_mouth = np.array(rigs)[:, 20:][::10]
    for e, folder in enumerate(folders):
        print(f'[{e}/{len(folders)}]:', folder)
        imgnames = os.listdir(os.path.join(root, folder))
        try:
           imgnames.sort(key=lambda x:int(x.split('.')[0]))
        except:
            continue
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        avaible_folders.append(folder)
        
        for i, imgname in tqdm(enumerate(imgnames)):
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[20:], axis=1)
            cnt_similar = np.sum(dist<0.5) 
            sample_freq = np.log(coff_count/cnt_similar)
            if sample_freq < 1:
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
        # break
    rigs = np.array(rigs)
    print('avaible folder: ', avaible_folders)
    print('total data:', len(data_list))
    with open(save_path, 'w') as f:
        f.writelines(data_list)

def check_is_faceware2render_paried():
    faceware_root = '/project/qiuf/DJ01/L36/faceware'
    render_root = '/project/qiuf/DJ01/L36/images'
    folders = os.listdir(faceware_root)
    save_root = '/project/qiuf/DJ01/L36/temp'

    for i, fold in enumerate(folders):
        faceware_fold = os.path.join(faceware_root, fold)
        render_fold = os.path.join(render_root, fold)
        save_fold = os.path.join(save_root, fold)
        os.makedirs(save_fold, exist_ok=True)
        if os.path.exists(faceware_fold) and os.path.exists(render_fold):
            imgnames = os.listdir(faceware_fold)
            print(f'{i}/{len(folders)}')
            for imgname in tqdm(imgnames,total=len(imgnames)):
                imgpath_faceware = os.path.join(faceware_fold, imgname)
                imgpath_render = os.path.join(render_fold, '{}.0000.jpg'.format(int(imgname.split('.')[0][-5:])))
                imgpath_saved = os.path.join(save_fold, imgname)
                if os.path.exists(imgpath_saved):
                    continue
                img_faceware= cv2.imread(imgpath_faceware)
                img_render = cv2.imread(imgpath_render)
                try:
                    img = np.concatenate((cv2.resize(img_faceware, (256,256)), cv2.resize(img_render, (256,256))), axis=0)
                    cv2.imwrite(imgpath_saved, img)
                except:
                    continue

def update_action_list_for_training():
    old_path = '/project/qiuf/DJ01/L36/images_test_list_actions.txt'
    images_root = '/project/qiuf/DJ01/L36/images'
    with open(old_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in tqdm(lines,total=len(lines)):
        if not os.path.exists(os.path.join(images_root, line.strip())):
            imgname = line.split('/')[-1]
            line = line.replace('/'+imgname, '_old/'+imgname)
        new_lines.append(line)
    with open(old_path.replace('.txt', '_new.txt'),'w') as f:
        f.writelines(new_lines)
    return

def format_images():
    root = '/project/qiuf/DJ01/L36_230/images/230_second'
    txtfiles = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]


    for txtfile in tqdm(txtfiles, total=len(txtfiles)):
        try:
            index = int(txtfile.split('/')[-1].split('.')[0])
        except:
            print(txtfile)
            break
        src = txtfile
        dst = os.path.join(os.path.dirname(src), f'{index:06d}.0000.jpg')
        os.rename(src, dst)
    return

def add_new_data_to_transet():  
    root = r'/project/qiuf/DJ01/L36_230_61/images'
    folders = os.listdir(root)
    data_list = []
    folders = ['ziva_L36_230_61_processed', 'ziva_L36_230_61_processed','ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 
               'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed','ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed']
    with open('/project/qiuf/DJ01/L36_230_61/images_train_list_actions321.txt', 'r') as f:
        imgs_list = f.readlines()
    for fold in folders:
        imgs = [y for x in os.walk(os.path.join(root, fold)) for y in glob.glob(os.path.join(x[0], '*.png'))]
        imgs_list += [l.replace(root+'/', '')+'\n' for l in imgs]
    with open('/project/qiuf/DJ01/L36_230_61/images_train_list_actions322.txt', 'w') as f:
        f.writelines(imgs_list)
    return

if __name__ == '__main__':
    import torchvision.transforms.transforms as transforms
    # format_images()
    # exit()
    # filter_image_data_with_actions_233_clean()
    # add_new_data_to_transet()
    load_rigs_to_cache('/project/qiuf/DJ01/L36/rigs', n_rig=61, version_old=666)
    # exit()
    # filter_image_data_with_actions_230_clean()
    exit()
    # update_action_list_for_training()
    # exit()
    # root = '/project/qiuf/L36_drivendata'
    # images = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    # print(len(images))
    transform2 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_path = '/project/qiuf/DJ01/L36/images'

    train_dataset = ABAWDataset2(root_path=data_path, character='l36', data_split='test', transform=transform2, random_flip=False, use_ldmk=False)

    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        )

    for i, inputs in enumerate(data_loader):
        print(i, inputs)
        if i > 50:
            break
