import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
import sys
sys.path.append("..")
from utils import get_msg_mgr


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        if not paths:
            raise ValueError('Each input data should have at least one element.')

        if all(pth.endswith('.pkl') for pth in paths):
            data_list = []
            for pth in paths:
                with open(pth, 'rb') as f:
                    data = pickle.load(f)
                data_list.append(data)
            for idx, data in enumerate(data_list):
                if len(data) != len(data_list[0]):
                    raise ValueError(
                        'Each input data({}) should have the same length.'.format(paths[idx]))
                if len(data) == 0:
                    raise ValueError(
                        'Each input data({}) should have at least one element.'.format(paths[idx]))
            return data_list

        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        if all(pth.lower().endswith(image_exts) for pth in paths):
            import cv2
            import numpy as np
            frames = []
            for pth in paths:
                img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f'Failed to read image: {pth}')
                frames.append(img)
            return [np.stack(frames, axis=0)]

        raise ValueError('Unsupported input file types in paths.')

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None
        
        # 数据结构类型: 'identity' (身份/类型/视角) 或 'class' (类别/身份/视角)
        data_structure = data_config.get('data_structure', 'identity')
        
        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        
        # 根据数据结构类型解析目录
        if data_structure == 'class':
            # 格式: 类别名/身份ID/视角 或 类别名/身份ID/样本/视角
            class_folders = sorted(os.listdir(dataset_root))
            class_pid_map = {}
            for cls in class_folders:
                cls_path = osp.join(dataset_root, cls)
                for pid in sorted(os.listdir(cls_path)):
                    class_pid_map[pid] = cls  # 身份ID映射到类别
            
            # 使用身份ID作为标签，类别信息存储在单独的字段
            label_list = []
            for cls in class_folders:
                cls_path = osp.join(dataset_root, cls)
                for pid in sorted(os.listdir(cls_path)):
                    label_list.append(pid)
            label_list = sorted(list(set(label_list)))
            
            # 构建标签到类别的映射
            self.label_to_class = {pid: class_pid_map[pid] for pid in label_list}
        else:
            # 默认格式: 身份ID/类型/视角/pkl
            label_list = os.listdir(dataset_root)
            self.label_to_class = None
        
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()
        
        # DEBUG: 打印关键信息
        msg_mgr.log_info(f"[DEBUG] data_structure: {data_structure}")
        msg_mgr.log_info(f"[DEBUG] dataset_root: {dataset_root}")
        msg_mgr.log_info(f"[DEBUG] partition labels: {partition.get('TRAIN_SET', [])}")
        msg_mgr.log_info(f"[DEBUG] label_list (from dirs): {label_list[:10]}...")
        msg_mgr.log_info(f"[DEBUG] filtered train_set: {train_set}")

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                if data_structure == 'class':
                    # 格式: 类别名/身份ID/视角 或 类别名/身份ID/样本/视角
                    cls = self.label_to_class[lab]
                    lab_path = osp.join(dataset_root, cls, lab)
                    if not osp.isdir(lab_path):
                        continue

                    first_level = sorted(os.listdir(lab_path))
                    first_level_dirs = [d for d in first_level if osp.isdir(osp.join(lab_path, d))]
                    if not first_level_dirs:
                        msg_mgr.log_debug('Find no data in %s-%s.' % (lab, cls))
                        continue

                    # 判断是 视角层 还是 样本层
                    def has_files(path):
                        return any(osp.isfile(osp.join(path, f)) for f in os.listdir(path))

                    is_view_level = any(has_files(osp.join(lab_path, d)) for d in first_level_dirs)

                    if is_view_level:
                        # 类别/身份/视角/帧
                        for view_name in first_level_dirs:
                            view_path = osp.join(lab_path, view_name)
                            view_files = sorted(os.listdir(view_path))
                            if view_files:
                                view_files = [osp.join(view_path, f) for f in view_files]
                                if data_in_use is not None:
                                    view_files = [dir for dir, use_bl in zip(
                                        view_files, data_in_use) if use_bl]
                                if view_files:
                                    seqs_info_list.append([lab, cls, view_name, view_files])
                    else:
                        # 类别/身份/样本/视角/帧
                        for sample_name in first_level_dirs:
                            sample_path = osp.join(lab_path, sample_name)
                            for view_name in sorted(os.listdir(sample_path)):
                                view_path = osp.join(sample_path, view_name)
                                if not osp.isdir(view_path):
                                    continue
                                view_files = sorted(os.listdir(view_path))
                                if view_files:
                                    view_files = [osp.join(view_path, f) for f in view_files]
                                    if data_in_use is not None:
                                        view_files = [dir for dir, use_bl in zip(
                                            view_files, data_in_use) if use_bl]
                                    if view_files:
                                        seqs_info_list.append([lab, cls, view_name, view_files])
                else:
                    # 默认格式: 身份ID/类型/视角/帧
                    for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                        for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                            seq_info = [lab, typ, vie]
                            seq_path = osp.join(dataset_root, *seq_info)
                            seq_dirs = sorted(os.listdir(seq_path))
                            if seq_dirs != []:
                                seq_dirs = [osp.join(seq_path, dir)
                                            for dir in seq_dirs]
                                if data_in_use is not None:
                                    seq_dirs = [dir for dir, use_bl in zip(
                                        seq_dirs, data_in_use) if use_bl]
                                seqs_info_list.append([*seq_info, seq_dirs])
                            else:
                                msg_mgr.log_debug(
                                    'Find no data in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
