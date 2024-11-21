import torch
import clip
from PIL import Image
import sys
sys.path.append("..")  # last level
import argparse
# from utils.utils_ import  template_prompting
import os.path as osp
# import av
# import decord
import numpy as np
import glob
import time
model_name_dict = {"ViT-B/16-lnpre": 'ViT-B-16.pt'}


def template_prompting(file, template = 'a photo of {}'):
    # class_names = []
    text_prompts = []

    for line in open(file):
        class_name = line.strip('\n')
        text_prompt = template.format(class_name)
        text_prompts.append(text_prompt)
        # class_names.append(line.strip('\n')

    return text_prompts

def read_mapping(file_):
    id_to_class_dict = dict()
    class_to_id_dict = dict()
    for line_id, line in enumerate(open(file_)):
        class_name = line.strip('\n')
        id_to_class_dict.update({line_id: class_name})
        class_to_id_dict.update({class_name: line_id})
    return id_to_class_dict, class_to_id_dict

def get_img_list(img_dir, format = '.png' ):
    img_list = glob.glob(osp.join( img_dir, f'*{format}'))
    img_list = sorted(img_list)
    return img_list

def get_args( setting = 'vit_b_16'):
    # env_id = get_env_id()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy',default=None)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.log_time = time.strftime("%Y%m%d_%H%M%S")
    if setting == 'vit_b_16':
        model_name = model_name_dict["ViT-B/16-lnpre"]
    else:
        raise Exception(f'Unknown setting {setting}')

    args.backbone_path = '/home/eicg/action_recognition_codes/home_ivanl_nvcluster/data/DeepInversion_results/train_models/EVL_models/CLIP_models'
    args.backbone_path = osp.join(args.backbone_path, model_name)

    args.class_file = '/media/data_8T/clip_depth_map_classification/clip_depth_map_classification/classes_all.txt'
    args.img_dir = '/media/data_8T/clip_depth_map_classification/clip_depth_map_classification/images'
    args.format = '.png'
    args.prompt_template = 'a depth map of {}'
    return args

def run(args):
    device = args.device
    top_k = args.top_k
    model, preprocess = clip.load(args.backbone_path, device=args.device)
    text_prompt_list = template_prompting(args.class_file, template=args.prompt_template)
    id_to_class_dict, class_to_id_dict = read_mapping(args.class_file)
    # text features
    text_tokenized = clip.tokenize(text_prompt_list).to(args.device)  # (n_cls, 77)
    # with torch.no_grad():
    #     text_features = model.encode_text(text_tokenized)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    img_list = get_img_list(img_dir=args.img_dir, format=args.format)
    n_imgs_total = len(img_list)
    print(f'{n_imgs_total} imags in total.' )
    n_batches = int(np.ceil(float(n_imgs_total) / args.batch_size))
    scores_all_imgs = torch.tensor([]).to(device)
    f_write = open(osp.join(args.result_dir, f'{args.log_time}_{args.prompt_template}_top{args.top_k}.txt'), 'w+')
    for batch_id in range(n_batches):
        start_point = batch_id*args.batch_size
        end_point = min( (batch_id+1) *args.batch_size, n_imgs_total )
        sample_id_list = list(range( start_point, end_point  ))
        n_samples_in_batch = len(sample_id_list)
        image_tensor_list = []
        for sample_id in sample_id_list:
            img_path = img_list[sample_id]
            pil_image = Image.open(img_path)
            processed_img = preprocess(pil_image).unsqueeze(0).to(device)
            image_tensor_list.append(processed_img)
        image_batch = torch.cat(image_tensor_list, dim = 0)


        with torch.no_grad():
            # image features
            # image_features = model.encode_image(image_batch)
            logits_per_image, _ = model(image_batch, text_tokenized)
            logits_per_image = logits_per_image.softmax(dim=-1)  # (bz, n_class )
        logits_per_image = logits_per_image.detach().cpu().numpy()
        for idx, sample_id in enumerate(sample_id_list):
            img_path = img_list[sample_id]
            img_filename = img_path.split('/')[-1].split('.')[0]

            img_score = logits_per_image[idx, :]
            top_k_indices = np.argpartition(img_score, -top_k)[-top_k:]
            top_k_classes = [id_to_class_dict[ind] for ind in list(top_k_indices)]
            top_k_scores = img_score[top_k_indices]
            sum_top_k_scores = np.sum(top_k_scores)
            # sort the top k scores
            sort_ind = np.argsort(-top_k_scores)
            top_k_scores_sorted = top_k_scores[sort_ind]
            top_k_classes_sorted = [top_k_classes[ind] for ind in list(sort_ind)]
            str_to_print = [f'{top_k_classes_sorted[idx]} {top_k_scores_sorted[idx]:.2f}' for idx in range(top_k)]
            str_to_print = f'{img_filename} {str_to_print}'
            print(str_to_print)
            f_write.write(f'{str_to_print}\n')

        # scores_all_imgs = torch
        print(f'Batch {batch_id} / {n_batches}')

    f_write.close()




if __name__ == '__main__':
    args = get_args()
    args.class_file = '/media/data_8T/clip_depth_map_classification/clip_depth_map_classification/classes_all.txt'
    args.img_dir = '/media/data_8T/clip_depth_map_classification/clip_depth_map_classification/images'
    args.backbone_path = '/home/eicg/action_recognition_codes/home_ivanl_nvcluster/data/DeepInversion_results/train_models/EVL_models/CLIP_models/ViT-B-16.pt'
    args.format = '.png'
    args.batch_size = 128
    args.result_dir = '/media/data_8T/clip_depth_map_classification/clip_depth_map_classification/prediction'
    args.top_k = 10
    args.prompt_template = 'a grayscale picture of {}'
    run(args)
