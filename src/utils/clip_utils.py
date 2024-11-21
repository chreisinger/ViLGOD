import torch
import clip
import time
import numpy as np

from pathlib import Path


# https://github.com/openai/CLIP
class ClipWrapper:
    def __init__(self, clip_cfg, model_path, device=None) -> None:
        assert model_path is not None, 'model_path is None'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.top_k = clip_cfg.top_k
        self.split_size = clip_cfg.split_size
        self.model, self.preprocess = clip.load(Path(model_path) / clip_cfg.model_name, device=device)
        self.template = clip_cfg.prompt_template
            
        self.id_to_class_dict = {idx: class_name for idx, class_name in enumerate(clip_cfg.class_list)}
        text_prompt_list = [clip_cfg.prompt_template.format(x) for x in clip_cfg.class_list]
        self.text_tokenized = clip.tokenize(text_prompt_list).to(device)
        self.text_features = self.model.encode_text(self.text_tokenized)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
    
    def __del__(self):
        del self.model
        del self.preprocess
        del self.text_tokenized
        del self.text_features

    def predict_clip_labels(self, images):
        image_splits = [self.preprocess(pil_image).unsqueeze(0).to(self.device) for pil_image in images]
        image_splits = torch.cat(image_splits, dim=0)
        image_splits = torch.split(image_splits, self.split_size)
        logit_list = []
        for image_split in image_splits:
            with torch.no_grad():
                frame_features = self.model.encode_image(image_split)
                frame_features /= frame_features.norm(dim=-1, keepdim= True)
                logits_per_image = (100.0 * frame_features @ self.text_features.T).softmax(dim=-1)
                logit_list.append(logits_per_image.detach().cpu())
                
        logits_per_image = torch.cat(logit_list, dim=0)
        logits_per_image = logits_per_image.numpy()
        
        cls_result_list = []
        score_result_list = []
        for idx in range(len(images)):
            img_score = logits_per_image[idx, :]
            top_k_indices = np.argpartition(img_score, -self.top_k)[-self.top_k:]
            top_k_classes = [self.id_to_class_dict[ind] for ind in list(top_k_indices)]
            top_k_scores = img_score[top_k_indices]
            # sort the top k scores
            sort_ind = np.argsort(-top_k_scores)
            top_k_scores_sorted = top_k_scores[sort_ind]
            top_k_classes_sorted = [top_k_classes[ind] for ind in list(sort_ind)]
            score_result_list.extend(top_k_scores_sorted)
            cls_result_list.extend(top_k_classes_sorted)
            
        return cls_result_list, score_result_list