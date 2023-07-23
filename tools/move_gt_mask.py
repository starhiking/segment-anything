import os 
import shutil
import cv2
import numpy as np

foodseg_mask_dir = "/home/jianghanyu/code/food_sam/segment-anything/dataset/FoodSeg103/Images/ann_dir"
uec_mask_dir = "/home/jianghanyu/code/food_sam/segment-anything/dataset/UECFoodPIXCOMPLETE"
base_dir = "/mnt/datasets/SAM_FOODSEG_NPY_NEW_DATASET_V2/"
dataset = ['FoodSeg103', 'UECFOODPIXCOMPLETE']
folder = ['test','train']

def visualization_save(mask_path, save_path, img_path):
    gt_mask = cv2.imread(mask_path)
    values = set(gt_mask.flatten().tolist())
    final_masks = []
    for v in values:
        if v != 0 :
            final_masks.append(gt_mask[:,:,-1] == v)
    final_masks = np.array(final_masks)
    np.random.seed(42)
    if len(final_masks) == 0:
        return
    h, w = final_masks[0].shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8) 
    for m in final_masks:
        color = np.random.randint(0,255, (3,))
        result[m, :] = color 
    image = cv2.imread(img_path)
    vis = cv2.addWeighted(image, 0.5, result, 0.5, 0) 
    cv2.imwrite(save_path, vis)


for d in dataset:
    for f in folder:
        final_save_dir = os.path.join(base_dir, d, f) #/mnt/datasets/SAM_FOODSEG_NEW_DATASET/FoodSeg103/train
        name_list = os.listdir(final_save_dir)
        if d == "FoodSeg103":
            ori_mask_dir = os.path.join(foodseg_mask_dir, f)
        else:
            ori_mask_dir = os.path.join(uec_mask_dir, f, 'mask')

        for name in name_list:
            if name == 'sam_process.log':
                continue
            save_path = os.path.join(final_save_dir, name, 'gt_mask.png')
            ori_mask_path = os.path.join(ori_mask_dir, name+'.png')
            shutil.copyfile(ori_mask_path, save_path)

            vis_save_path = os.path.join(final_save_dir, name, 'gt_vis.png')
            img_path = os.path.join(final_save_dir, name, 'input.jpg')
            print(vis_save_path)
            visualization_save(ori_mask_path, vis_save_path, img_path)
