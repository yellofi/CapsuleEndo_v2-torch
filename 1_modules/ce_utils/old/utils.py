#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import numpy as np
import glob
import cv2
from itertools import product

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
#     percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {} / {}'.format(prefix, bar, iteration, suffix)),
#     sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def Generate_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)     

def target_preprocessings(phase_a_switch = [1, 1, 1], phase_b_switch = True, mode = 'load'):
    """
    phase_a_switch = [1, 1, 1], [0, 0 ,1], [1, 1, 0].... 
    that means [flip, rotate, blur_sharp]
    """
    phase0 = ['_c']
    phase1 = {1: ['-', 'f'], 0: ['-']}
    phase2 = {1: ['-', 'r1', 'r2', 'r3'], 0: ['-']}
    phase3 = {1: ['-', 'ab', 'mb', 'eh'], 0: ['-']}
    phase4 = ['s_-30_v_30', 's_-30_v_-30', 's_30_v_-30', 's_30_v_30']

    if mode == 'load':
        phase_a_items = [phase1[phase_a_switch[0]], phase2[phase_a_switch[1]], phase3[phase_a_switch[2]]]
    elif mode == 'preprocessing':
        phase_a_items = [phase0, phase1[phase_a_switch[0]], phase2[phase_a_switch[1]], phase3[phase_a_switch[2]]]

    phase_a = []
    for i in list(product(*phase_a_items)):
        phase_a.append('_'.join(i))

    if not phase_b_switch != True:
        phase_b = []
        for i in list(product(*[phase_a, phase4])):
            phase_b.append('_'.join(i))
        return list(np.hstack([phase_a, phase_b]))
    else:
        return phase_a 
    
# Preprocessing
class ce_preprocessing:
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir

    def cropping(self, img):
        img = np.array(img, dtype = 'f4')
        img_pre = img[32:544, 32:544, :]
        for i in range(100):
            for j in range(100):
                if i + j > 99:
                    pass
                else :
                    img_pre[i, j, :] = 0
                    img_pre[i, -j, :] = 0
        return img_pre.astype('uint8')

    def rotate(self, img, degree):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D(center = (cols/2, rows/2), angle = degree, scale = 1)
        img_rotated = cv2.warpAffine(img, M, dsize = (rows, cols))
        return img_rotated
    
    def blur_and_sharp(self, img):
        img_avg_blur = cv2.blur(img, (5,5)).astype('uint8')
        
        kernel_size = 15
        
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        img_mb = cv2.filter2D(img, -1, kernel_motion_blur).astype('uint8')
        
        kernel_edge_enhancement = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
        img_eh = cv2.filter2D(img, -1, kernel_edge_enhancement).astype('uint8')    
        return img_avg_blur, img_mb, img_eh
    
    def bgr2_h_s_v(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return h, s, v

    def hsv_control(self, ch_data, ctr_value, ch_name):
        """
        ch_data: data of channel (h, s, or v) which you want to revise by ctr_value / shape: image.shape[0:2]
        ctr_value: the value that will be added to corresponding channel.
        ch_name: 'h', 's', or 'v'
        """
        ch_data_rev = ch_data.copy()
        if ctr_value > 0:
            ch_data_rev[np.where(ch_data <= 255 - ctr_value)] += ctr_value
        else:
            ch_data_rev[np.where(ch_data + ctr_value >= 0)] -= abs(ctr_value)
        return ch_data_rev
    
    def pre_aug(self, img, phase = 'x160'):  
        """
        The image will be preprocessed and augmented at one go 
        by an entire process consisting of  the repetitive statement (for loop) per the processing phase 
        """
        preprocessed_imgs = []
        preprocessed_nots = []
        
        crop = self.cropping(img)
        if phase == 'crop':
            return [crop], ['_c_-_-_-']
        else:
            sv_ctr_values = [-30, 30]
            c_r1, c_r2, c_r3 = self.rotate(crop, 90), self.rotate(crop, 180), self.rotate(crop, 270)
            for r, r_n in zip([crop, c_r1, c_r2, c_r3], ['-', 'r1', 'r2', 'r3']):
                r_f = np.flipud(r)
                for f,  f_n in zip([r, r_f], ['-', 'f']): 
                    f_ab, f_mb, f_edge = self.blur_and_sharp(f)
                    for b, b_n in zip([f, f_ab, f_mb, f_edge], ['-', 'ab', 'mb', 'eh']):                    
                        preprocessed_imgs.append(b)
                        not_ = '_c_{}_{}_{}'.format(f_n, r_n, b_n)
                        preprocessed_nots.append(not_)
                        h, s, v = self.bgr2_h_s_v(b)
                        for s_value in sv_ctr_values:
                            s_rev = self.hsv_control(s, s_value, ch_name = 's')
                            for v_value in sv_ctr_values:
                                v_rev = self.hsv_control(v, v_value, ch_name = 'v')
                                v_rev[np.where(v <= 7)] = 0
                                b_sv = cv2.merge((h, s_rev, v_rev))
                                b_sv = cv2.cvtColor(b_sv, cv2.COLOR_HSV2BGR)
                                preprocessed_imgs.append(b_sv)
                                not_ = '_c_{}_{}_{}_s_{}_v_{}'.format(f_n, r_n, b_n, s_value, v_value)
                                preprocessed_nots.append(not_) 
                if not phase != 'before_rotation':
                    break
            return preprocessed_imgs, preprocessed_nots
    
    def avg_blur(self, img):
        return cv2.blur(img, (5,5)).astype('uint8')
    
    def motion_blur(self, img):
        kernel_size = 15
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        return cv2.filter2D(img, -1, kernel_motion_blur).astype('uint8')
    
    def edge_enhancement(self, img):
        kernel_edge = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
        return cv2.filter2D(img, -1, kernel_edge).astype('uint8') 
    
    def s_rev(self, img, s_value):
        h, s, v = self.bgr2_h_s_v(img)
        s_rev = self.hsv_control(s, s_value, ch_name = 's')
        return [h, s_rev, v]
    def v_rev_after_s_rev(self, s_rev_outputs, v_value):
        h, s_rev, v = s_rev_outputs
        v_rev = self.hsv_control(v, v_value, ch_name = 'v')
        v_rev[np.where(v <= 7)] = 0
        img_sv = cv2.merge((h, s_rev, v_rev))
        return cv2.cvtColor(img_sv, cv2.COLOR_HSV2BGR)
            
    def pre_aug_target_phase(self, img, phase = 'c'): 
        
        """
        phase, ex) 'c_f_-_mb_s_-30_v_30' -> 'c_f_-_mb_s-30_v30' -> ['c', 'f', '-', 's-30','v30']
        It allows to preprocess the image in specific phase, but slower it is fit to check preprocessing with small data
        """
        function = {'': (lambda x: x), '-': (lambda x: x),
                    'c': (lambda x: self.cropping(x)),
                    'f': (lambda x: np.flipud(x)), 
                    'r1': (lambda x: self.rotate(x, 90)), 
                    'r2': (lambda x: self.rotate(x, 180)), 
                    'r3': (lambda x: self.rotate(x, 270)),
                    'ab': (lambda x: self.avg_blur(x)),
                    'mb': (lambda x: self.motion_blur(x)),
                    'eh': (lambda x: self.edge_enhancement(x)),
                    's-30': (lambda x: self.s_rev(x, -30)),
                    's30': (lambda x: self.s_rev(x, 30)),
                    'v-30': (lambda x: self.v_rev_after_s_rev(x, -30)),
                    'v30': (lambda x: self.v_rev_after_s_rev(x, 30))}
        values = ['-30', '30']
        for i in values:
            if i in phase:
                phase = phase.replace('_{}'.format(i), str(i))
        phase_seg = phase.split('_')  
        for i, p in zip(range(len(phase_seg)), phase_seg):
            if i == 0:
                p_img = function[p](img)
            else:
                p_img = function[p](p_img)
        return p_img
    
    def pre_aug_and_save(self, phase, cls, les, filename, preprocessing_phase = 'x160', pre_aug_type = 'for_loop',
                         phase_a = [1, 1, 1], phase_b = True):
        
        """
        phase = 'train', 'test'
        cls: [les]  
          'n': ['neg']
          'h': ['redspot', 'angio', 'active'], 
          'd': ['ero', 'ulc', 'str'],
          'p': ['amp', 'lym', 'tum']}
        preprocessing_phase = 'x160', 'crop', 'before_rotation' for pre_aug
        phase_a = [1, 1, 1], [1, 0, 1], [1, 1, 0] .... [flip, rotate, blur_sharp]
        phase_b = True -> phase_a (max. x32) + phase_a * sv_control (max. x32x4) => max, 32 x 5
        """
        lesions = dict(neg = 'negative', 
                       redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                       ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                       amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
        classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')
        
        save_path = os.path.join(self.save_dir, phase, classes[cls], lesions[les])
        import_path = os.path.join(self.data_dir, classes[cls], lesions[les])

        if not(os.path.isdir(save_path)):
            os.makedirs(save_path)
        
        for i, f in zip(range(1, len(filename)+1), filename) :
            img = cv2.imread(import_path + '/' + f)
            if pre_aug_type == 'for_loop':
                p_imgs, p_nots = self.pre_aug(img, phase = preprocessing_phase)  
                for img_, not_ in zip(p_imgs, p_nots):
                    save_filename = os.path.join(save_path, '{}_{}{}'.format(f[:-4], not_, f[-4:]))
                    if not(os.path.isfile(save_filename)):
                        cv2.imwrite(save_filename, img_)
            elif pre_aug_type == 'target_phase':
                for not_ in target_preprocessings(phase_a, phase_b, mode = 'preprocessing'):
                    save_filename = os.path.join(save_path, '{}_{}{}'.format(f[:-4], not_, f[-4:]))
                    if not(os.path.isfile(save_filename)):
                        p_img = self.pre_aug_target_phase(img, phase = not_)
                        cv2.imwrite(save_filename, p_img)
            printProgress(i, len(filename), prefix = '{:05d}'.format(len(filename)))       
        return

def Reshape4torch(img):
    """
    (sample #, height, width, channel) -> (sample #, channel, height, width)
    """
    if len(img.shape) == 4:
        img = np.transpose(img, (0, 3, 1, 2))
        return img
    elif len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)
    
def GenerateLabel(data, cls):
#     label = cls*np.ones([data.shape[0]])
    label = cls*np.ones([len(data)])
    return label


# Load data
def load_data(phase, cls, les = None, data = 'sm', data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed', 
              extract_name = False, image_ch = 'bgr'):
    """
    phase = 'train', 'test'
    cls: [les]  
      'n': ['neg']
      'h': ['redspot', 'angio', 'active'], 
      'd': ['ero', 'ulc', 'str'],
      'p': ['amp', 'lym', 'tum']}
    """
    lesions = dict(neg = 'negative', 
                   redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                   ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                   amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
    classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')

    path = os.path.join(data_dir, data, phase, classes[cls], lesions[les])
    pathlist = glob.glob(path + '/*.jpg')

    return load_image_from_path(pathlist, image_ch = image_ch, extract_name = extract_name)

def load_image_from_path(pathlist,image_ch = 'bgr', extract_name = False):
    data = []
    for i in pathlist:
        temp = cv2.imread(i)
        if image_ch == 'bgr':
            pass
        elif image_ch == 'rgb':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        elif image_ch == 'hsv':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        data.append(temp)
    if extract_name != False:
        name = []
        for i in pathlist:
            name.append(os.path.basename(i))
        return Reshape4torch(np.asarray(data)), np.asarray(name)
    else:
        return Reshape4torch(np.asarray(data))

    
class sm160_dataset:
    def __init__(self, phase, data, pre_a, pre_b, img_ch = 'bgr', ext_name = True):
        self.phase = phase        # 'train' or 'test'
        self.data = data          # 'sm', 'sm_core', 'sm_v2', 'sm_x160', ...
        self.pre_a = pre_a        # [1, 1, 1], [0, 0 ,1], [1, 1, 0].... 
        self.pre_b = pre_b        # True or False
        self.img_ch = img_ch      # 'bgr', 'rgb', and 'hsv'
        self.ext_name = ext_name  # True or False

    def load_path(self, cls, les, data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'):
        """
        phase = 'train', 'test'
        cls: [les]  
          'n': ['neg']
          'h': ['redspot', 'angio', 'active'], 
          'd': ['ero', 'ulc', 'str'],
          'p': ['amp', 'lym', 'tum']}
        pre_a[0] must be 0
        """
        self.lesions = dict(neg = 'negative', 
                       redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                       ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                       amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
        self.classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')

        path = os.path.join(data_dir, self.data, self.phase, self.classes[cls], self.lesions[les])
        pathlist = glob.glob(path + '/*.jpg')
        

        path_in_phase = []
        for p in pathlist:
            name = os.path.basename(p)
            if (name.split('c_')[-1])[:-4] in target_preprocessings(self.pre_a, self.pre_b):
                path_in_phase.append(p)   
        return np.asarray(path_in_phase)

        
#         if self.pre_b != True:
#             path_in_phase = []
#             for p in pathlist:
#                 name = os.path.basename(p)
#                 if (name.split('c_')[-1])[:-4] in target_preprocessings(self.pre_a, self.pre_b):
#                     path_in_phase.append(p)   
#             return np.asarray(path_in_phase)
#         else:
#             return np.asarray(pathlist)
        
    def extract_origin_path(self, pathlist):
        crop_path = []
        for path in pathlist:
            name = os.path.basename(path)
            if name.split('__c_')[-1][:-4] == '-_-_-':
                crop_path.append(name.split('__c_')[:-1][0])
        return np.asarray(crop_path)

    def train_valid_path(self, name, pathlist, train_rate = 0.9):
        t_idx = np.random.choice(len(name), round(train_rate*len(name)), replace = False)
        v_idx = np.setdiff1d(np.arange(len(name)), t_idx)

        train_name = name[t_idx]
        valid_name = name[v_idx]

        train_path, train_aug_path = [], []
        valid_path = []

        for path in pathlist:
            filename = os.path.basename(path)
            if filename.split('__c_')[:-1] in train_name:
                train_aug_path.append(path)
                if filename.split('__c_')[-1][:-4] == '-_-_-':
                    train_path.append(path)
            elif filename.split('__c_')[:-1] in valid_name and filename.split('__c_')[-1][:-4] == '-_-_-':
                valid_path.append(path)
        return np.asarray(train_path), np.asarray(train_aug_path), np.asarray(valid_path)
    
    def Get_train_valid_set(self, cls = 'n', les = 'neg', train_rate = 0.9):
        # target preprocessing에 대해 augmented된 data의 file경로를 가져온다.
        total_aug_path = self.load_path(cls, les)
        # 원본 파일 이름 추출
        origin_path = self.extract_origin_path(total_aug_path)
        # 원본 이름을 통해 9:1로 training set과 validation set을 나누는데,
        # 훈련 중 학습을 위한 augmented train 데이터 파일이름을 target preprocessing에 대해 증강된 만큼 가져와
        # 훈련 중에 미니 배치만큼 로드하여 모델 업데이트에 사용 (증강된 데이터를 다 불러놓고 쓰는 것은 메모리에 안 좋음)
        # 훈련 중 평가를 위해 필요한 원본 train_x, valid_x를 전체 로드함
        train_path, train_aug_path, valid_path = self.train_valid_path(origin_path, total_aug_path, train_rate = 0.9)
#         print(train_path)
#         print(valid_path)
        train_x = load_image_from_path(train_path, image_ch = self.img_ch, extract_name = self.ext_name)
        valid_x = load_image_from_path(valid_path, image_ch = self.img_ch, extract_name = self.ext_name) 
#         print(type(valid_x))
#         print(len(valid_x))
#         print(valid_x)
        if type(valid_x) == type(None):
            print(self.classes[cls], '-', self.lesions[les], '| augmented training data:', len(train_aug_path), 
                  '|(For validation) train x:', train_x.shape, 'valid x:', valid_x)
        else:
            print(self.classes[cls], '-', self.lesions[les], '| augmented training data:', len(train_aug_path), 
                  '|(For validation) train x:', train_x.shape, 'valid x:', valid_x.shape)
        return train_aug_path, train_x, valid_x
         
    def load_data(self, cls, les):
        pathlist = self.load_path(cls, les)
        return load_image_from_path(pathlist, image_ch = self.img_ch, extract_name = self.ext_name)

# Training 
    
# For evaluation    
def Batch_idxs(data, batch_size = 250):
    """generate the serial batch of data on index-level.
       Usually, the data is too large to be evaluated at once.
    
    Args:
      data: A list or array of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch_idxs: A list, 
    """
    total_size = len(data)
    batch_idxs = []
    start = 0
    while True:
        if total_size >= start + batch_size:
            batch_idxs.append([start + i for i in range(batch_size)])
        elif total_size < start + batch_size:
            batch_idxs.append([start + i for i in range(total_size - start)])
        start += batch_size
        if total_size <= start:
            break
    return batch_idxs

def batch_idxs(dataset, batch_size = 32, shuffle = True):

    idxs = np.arange(len(dataset))
    total_size = len(idxs)
    if shuffle:
        np.random.shuffle(idxs)
    start = 0
    b_idxs = []
    while True:
        if total_size > start + batch_size: 
            b_idxs.append(list(idxs[start:start+batch_size]))  
            start += batch_size
        elif total_size <= start + batch_size: 
            b_idxs.append(list(idxs[start:]))
            break 
    return b_idxs

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    

def Sec_to_M_S_ms(sec):
    min_sec = time.strftime("%M:%S", time.gmtime(sec))
    ms = '{:03d}'.format(int((sec - int(sec))*1000))   
    return '.'.join([min_sec, ms])

def model_pred_and_time(model, image, model_type = 'binary'):
    b_idxs = test_batch_idxs(image)
    if model_type == 'ensemble':
        e_outputs, nh_outputs, nd_outputs = [], [], []
        start_time = time.time()
        for i, b_idx in enumerate(b_idxs):
            e_softmax, nh_softmax, nd_softmax = model.get_softmax(image[b_idx])
            e_outputs.append(e_softmax), nh_outputs.append(nh_softmax), nd_outputs.append(nd_softmax)
            printProgress(i+1, len(b_idxs), barLength = 80,
                          prefix = '# of batch: {}'.format(len(b_idxs)), 
                          suffix = 'model prediction ({})'.format(model_type[0]))
        time_taken = time.time() - start_time
        time_taken = Sec_to_M_S_ms(time_taken)
        return np.concatenate(e_outputs), np.concatenate(nh_outputs), np.concatenate(nd_outputs), time_taken
    else:
        outputs = []
        start_time = time.time()
        for i, b_idx in enumerate(b_idxs):
            softmax = model.get_softmax(image[b_idx])
            outputs.append(softmax)
            printProgress(i+1, len(b_idxs), barLength = 80,
                          prefix = '# of batch: {}'.format(len(b_idxs)), 
                          suffix = 'model prediction ({})'.format(model_type[0]))
        time_taken = time.time() - start_time
        time_taken = Sec_to_M_S_ms(time_taken)
        return np.concatenate(outputs), time_taken

