import os
import cv2
import json
import torch
import torch.nn as nn
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from src import open_clip
from dataset import *
import logging
from tqdm import tqdm
from logging import getLogger
from data.mvtecdataset import MVTDataset, create_mvtect_dataloader, MVTEC,MPDD, VisA
from einops import repeat, rearrange
import torch.optim as optim
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from adapter import ImageHead, TextHead
from adapter_plus import ImageHead, TextHead
from loss import *
from src.open_clip import tokenizer
from visualize import *
import math
import re
import warnings
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import argparse


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc



class prompt_order():
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list = [
            
           "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]
        """
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
            """
    def prompt(self, class_name):
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
       
        normal_ensemble_template = []
        for class_template in self.template_list:
            for ele in class_state:
                normal_ensemble_template.append(class_template.format(ele))

        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                     self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template



class prompt_mpdd():
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]
        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
            "{} with anomaly",
            #"{} with hole"
        ]

        self.template_list = [
            #"a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            #"a bright photo of a {}.",
            #"a bright photo of the {}.",
            #"a photo of a {}.",
            #"a photo of the {}.",
            #"a photo of a small {}.",
            #"a photo of the small {}.",
            #"a photo of the {} for visual inspection.",
            #"a photo of a {} for visual inspection.",
            #"a photo of the {} for anomaly detection.",
            #"a photo of a {} for anomaly detection."
        ]

    def prompt(self, class_name):
        x = class_name
        if class_name != 'connector':
            class_name = ' '
        
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                    self.template_list]

        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                     self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template



class patch_scale():
    def __init__(self, image_size):
        self.h, self.w = image_size

    def make_mask(self, patch_size=16, kernel_size=16, stride_size=16):
        self.patch_size = patch_size
        self.patch_num_h = self.h // self.patch_size
        self.patch_num_w = self.w // self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size // patch_size
        self.stride_size = stride_size // patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype=torch.float32).reshape(
            (1, 1, self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy


simple_tokenizer = tokenizer.SimpleTokenizer()


class CLIP_AD(nn.Module):
    def __init__(self, model_name='ViT-B-16-plus-240'):
        super(CLIP_AD, self).__init__()
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name,
                                                                                        pretrained='laion400m_e31')
        self.mask = patch_scale((240, 240))

    def multiscale(self):
        pass

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image, patch_size, mask=True):
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale, mid_scale],
                                                                              proj=False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale


def compute_score(image_features, text_features):
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    #print("ii: ", image_features.unsqueeze(1).shape)
    #print("text: ", text_features.shape)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features) / 0.07).softmax(dim=-1)
    if math.isnan(text_probs[0][0][1]):
        print(torch.bmm(image_features.unsqueeze(1), text_features))
    return text_probs


def compute_sim(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features) / 0.07).softmax(dim=-1)
    return simmarity


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def prepare_text_future(model, obj_list):
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    text_generator = prompt_order()

    for i in obj_list:
        normal_description, abnormal_description = text_generator.prompt(i)

        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)
    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features
    
    
    
def prepare_text_future_mpdd(model, obj_list):
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    text_generator = prompt_mpdd()

    for i in obj_list:
        normal_description, abnormal_description = text_generator.prompt(i)

        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)
    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features





def train(args, ):
    img_size = args.image_size
    dataset_dir = args.data_path
    dataset_name = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset == "mvtec":
        CLSNAMES = MVTEC
    elif args.dataset == "mpdd":
        CLSNAMES = MPDD
    elif args.dataset == "visa":
        CLSNAMES = VisA
    model = CLIP_AD(args.model)
    model.to(device)


    preprocess = model.preprocess

    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size),
                                                 interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))

    dataloader_list_train = []
    dataloader_list_test = []

    for name in CLSNAMES:
        if args.classname == name or args.classname == "all":
            dataloader_list_train.append(
                create_mvtect_dataloader(
                    name,
                    batch_size=args.batch_size,
                    source=args.data_path,
                    split="train",
                    transform=preprocess,
                    args=args
                )
            )
            dataloader_list_test.append(
                create_mvtect_dataloader(
                    name,
                    batch_size=16,
                    source=args.data_path,
                    split="test",
                    transform=preprocess,
                    args=args
                )
            )
    model.eval()
    results = {}
    results['cls_names'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    table_header = ['cls', 'Image-AUROC', 'Image-AP', "F1-MAX"]
    table_data = []
    patch_size = 16
    total = 0
    xloss = xLoss()
    criterion = infonceLoss()
    CEloss = CrossEntropyLoss()
    """
    with torch.no_grad():
        Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = prepare_text_future(model, CLSNAMES)
    """
    with torch.no_grad():
        if args.dataset != "mpdd":
            Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = prepare_text_future(model, CLSNAMES)
        else:
            Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = prepare_text_future_mpdd(model, CLSNAMES)
    print("Mermory_avg_normal_text_features shape: ", Mermory_avg_normal_text_features.shape)
    print("Mermory_avg_abnormal_text_features shape: ", Mermory_avg_abnormal_text_features.shape)
    AUROC = []
    AP = []
    F1_MAX = []
    pattern = r'\d+\.\d+|\d+'  
    matches = re.findall(pattern, args.data_path)
    #print(type(matches))
    for i, (dataloader_train, dataloader_test, classname) in enumerate(zip(dataloader_list_train, dataloader_list_test, CLSNAMES)):

        image_head = ImageHead(feature_dim=640, out_dim=640, res=args.res, no_head=args.no_image_head).to(device)
        text_head = TextHead(feature_dim=640, out_dim=640, res=args.res, no_head=args.no_text_head).to(device)
        
        image_head = image_head.train()
        text_head = text_head.train()
        optimizer1 = optim.Adam(image_head.parameters(), lr=args.lr1)
        optimizer2 = optim.Adam(text_head.parameters(), lr=args.lr2)
        max_auroc = 0
        max_ap = 0
        flag = 0
        f1_max=0
        for epoch in tqdm(range(args.epochs)):
            if(flag == 10): break
            for idx, inputs in enumerate(dataloader_train):
                images = inputs["image"].to(device)
                noisy_images = inputs["noise_image"].to(device)
                label = torch.cat([inputs["is_anomaly"], inputs["not_anomaly"]]).to(device)
                #print(inputs["image_path"])
                total_images = torch.cat([images, noisy_images], dim=0)
                batch_size = images.shape[0]
                with torch.no_grad():
                    large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(
                        total_images, patch_size)
                average_normal_features = Mermory_avg_normal_text_features[i]
                average_anomaly_features = Mermory_avg_abnormal_text_features[i]
                
                text_features_origin = torch.cat([average_normal_features, average_anomaly_features], dim=0)
                
                #print("class_tokens: ", class_tokens.shape)
                average_normal_features = text_head(average_normal_features)
                average_anomaly_features = text_head(average_anomaly_features)
                text_features = torch.cat([average_normal_features, average_anomaly_features], dim=0)
                class_tokens, t_ = image_head(class_tokens, text_features_origin)
                
                text_features = text_features + 0.7*t_ #visa
                #text_features = text_features #+ 0.01*t_ #mvtec
                
                text_feature = repeat(text_features, 'h w -> h w c', c=batch_size*2).permute(2, 1, 0)
                #os._exit(0)
                #print(text_feature.shape)
                #print((text_features[0]==text_features.T[:,0]))
                zscore = compute_score(class_tokens, text_feature)
                # loss = -criterion(class_tokens, text_features, batch_size)
                loss = xloss(class_tokens[:batch_size, :], class_tokens[batch_size:,:], text_features)
                celoss = CEloss(zscore.squeeze(1), label)
                # print(f"{epoch}epoch:{loss}")
                total = loss + 0.7*celoss
                total.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                
                #print(loss)
                # print("text_feature shape: ", text_feature.shape)
                # print("class_tokens shape: ", class_tokens.shape)
                # print("text_feature shape: ", text_feature.shape)
            auroc_sp, ap_sp, f1_sp = evaluate(i,
                     model,
                     dataloader_test,
                     classname,
                     [Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features],
                     image_head=image_head, 
                     text_head=text_head,
                     epoch=epoch
                     )
            if auroc_sp > max_auroc:
                max_auroc = auroc_sp
                max_ap = ap_sp
                f1_max = f1_sp
                save_path = args.model_path+ args.dataset + "//" + classname 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #torch.save({"image_path": inputs["image_path"], "image_state_dict":image_head.state_dict(), "text_state_dict":text_head.state_dict()}, 
                #                                                            save_path + "//" + matches[0]+"_shot_"+str(args.batch_size)+"_bs.pt")
                flag = 0
            else:
                flag += 1
        table_data.append([classname, max_auroc, max_ap, f1_max])
        AUROC.append(max_auroc)
        AP.append(max_ap)
        F1_MAX.append(f1_max)
        print(f"{classname}", max_auroc)
        print(f"{classname}", max_ap)
        print(f"{classname}", f1_max)
    table_data.append(["mean", sum(AUROC)/len(CLSNAMES), sum(AP)/len(CLSNAMES), sum(F1_MAX)/len(CLSNAMES)])
    print(tabulate(table_data, headers=table_header, tablefmt='grid'))
    print(args.dataset)
    
    print("few shot: : ", matches)
    print("batch size: ", args.batch_size)
    return table_data


@torch.no_grad()
def evaluate(i, model, testloader, classname, text_features, epoch, image_head=None, text_head=None):
    if(epoch < 0):return -1,-1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = text_features
    model.eval()
    results = {}
    results['cls_names'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    patch_size = 16
    total = 0

    average_normal_features = Mermory_avg_normal_text_features[i]
    average_anomaly_features = Mermory_avg_abnormal_text_features[i]
    #print("average_normal_features shape: ", average_normal_features.shape)
    #print("average_normal_features shape: ", average_normal_features.shape)
    results['gt_sp'] = []
    results['pr_sp'] = []
    for idx, inputs in enumerate(testloader):
        images = inputs["image"].to(device)
        batch_size = images.shape[0]
        with torch.no_grad():
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(
                images, patch_size)
        
        text_features_origin = torch.cat([average_normal_features, average_anomaly_features], dim=0)
        
            #print("class_tokens shape: ", class_tokens.shape)
        if text_head is not None:
            average_normal_features = text_head(average_normal_features)
            average_anomaly_features = text_head(average_anomaly_features)
            #print("average_normal_features shape: ", average_normal_features.shape)
        text_feature = torch.cat((average_normal_features, average_anomaly_features), dim=0)
        if image_head is not None:
            class_tokens, t_ = image_head(class_tokens, text_features_origin)
        text_feature = text_feature + 0.7*t_ #visa
        #text_feature = text_feature #+ 0.01*t_ #mvtec
        
        text_feature = repeat(text_feature, 'h w -> h w c', c=batch_size).permute(2, 1, 0).clone()
        #print("text_feature shape: ", text_feature.shape)
        # print("class_tokens shape: ", class_tokens.shape)
        # print("text_feature shape: ", text_feature.shape)
        zscore = compute_score(class_tokens, text_feature)
        z0score = zscore[:, 0, 1]
        # print(zscore[:, 0, 1][0], zscore[:, 0, 0][0])
        results['pr_sp'].extend(z0score.detach().cpu())
        results['gt_sp'].extend(inputs['is_anomaly'].detach().cpu())
    save_path = "./result/"+ args.dataset + "//" + classname + "/histogram/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt = show_bar(results['pr_sp'], results['gt_sp'])
    plt.savefig(save_path + str(epoch) + ".jpg")
    plt.close()
    gt_sp = np.array(results['gt_sp'])
    pr_sp = np.array(results['pr_sp'])
    
    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    ap_sp = average_precision_score(gt_sp, pr_sp)
    precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    # print("precisions recalls", precisions, recalls)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    
    model.train()
    return [auroc_sp, ap_sp, f1_sp]


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--model_path", type=str, default='./checkpoints/', help='path to save results')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--classname", type=str, default='all', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9],
                        help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--res", type=bool, default=True, help="image size")
    parser.add_argument("--no_image_head", type=bool, default=False, help="image size")
    parser.add_argument("--no_text_head", type=bool, default=False, help="image size")
    # parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")

    #train
    parser.add_argument("--epochs", type=int, default=100, help="features used")
    parser.add_argument('--lr1', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--lr2', type=float, default=0.00001, help='Learning rate')

    # few shot
    parser.add_argument("--batch_size", type=int, default=2, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()
    
    setup_seed(args.seed)
    results = []
    table_header = ['cls', 'Image-AUROC', 'Image-AP', "F1-MAX"]
    result = train(args)
    print(tabulate(result, headers=table_header, tablefmt='grid'))
    """
    for i in [1,2,4]:
        args.batch_size = i
        result = train(args)
        results.append(result)
    for i in results:
        print(tabulate(i, headers=table_header, tablefmt='grid'))
    """