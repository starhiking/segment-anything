import numpy as np
import cv2
import os
import csv
import logging

def create_logger():
    
    log_file = f"foodseg_cal_sam_masks_label.log"
    final_log_file = os.path.join("tools", log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger


def calculate_single_image_masks_label(mask_file, pred_mask_file, logger, category_list, new_mask_label_file_name):
    """
        每个子文件夹保存一种，注意命名，不然会覆盖
        保存的格式： mask_index, category_id, category_name, category_count
    """
    sam_mask_data = np.load(mask_file)
    pred_mask_img = cv2.imread(pred_mask_file)[:,:,-1] # red channel
    shape_size = pred_mask_img.shape[0] * pred_mask_img.shape[1]

    folder_path = os.path.dirname(pred_mask_file)
    sam_mask_category_folder = os.path.join(folder_path, "sam_mask_category_label")
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(sam_mask_category_folder, new_mask_label_file_name)
    with open(mask_category_path, 'w') as f:
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i]
            single_mask_labels = pred_mask_img[single_mask]
            unique_values, counts = np.unique(single_mask_labels, return_counts=True, axis=0)
            max_idx = np.argmax(counts)
            single_mask_category_label = unique_values[max_idx]
            count_ratio = counts[max_idx]/counts.sum()

            logger.info(f"{folder_path}/sam_mask/{i} assign label: [ {single_mask_category_label}, {category_list[single_mask_category_label]}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]")
            f.write(f"{i},{single_mask_category_label},{category_list[single_mask_category_label]},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n")

    f.close()


if __name__ == '__main__':

    # !!!!! must change when use different pred mask or gt mask !!!!!
    new_mask_label_file_name = "gt_semantic_masks_category.txt"
    pred_mask_file_name = "gt_mask.png"

    test_paths = ["SAM_FOODSEG_NPY_NEW_DATASET_V2/FoodSeg103/test","SAM_FOODSEG_NPY_NEW_DATASET_V2/UECFOODPIXCOMPLETE/test"]

    category_lists = []
    with open("tools/category_id_files/foodseg103_category_id.txt", 'r') as f:
        foodseg103_category_lines = f.readlines()
        foodseg103_category_list = [' '.join(line_data.split('\t')[1:]).strip() for line_data in foodseg103_category_lines]
        f.close()
        category_lists.append(foodseg103_category_list)
    with open("tools/category_id_files/UECFOODPIXCOMPLETE_category_id.txt", 'r') as f:
        UECFOODPIXCOMPLETE_category_lines = f.readlines()
        UECFOODPIXCOMPLETE_category_list = [' '.join(line_data.split('\t')[1:]).strip()  for line_data in UECFOODPIXCOMPLETE_category_lines]
        f.close()
        category_lists.append(UECFOODPIXCOMPLETE_category_list)
    
    logger = create_logger()

    for test_path, category_list in zip(test_paths, category_lists):
        img_ids = os.listdir(test_path)
        for img_id in img_ids:
            mask_file_path = os.path.join(test_path, img_id, "sam_mask/masks.npy")
            pred_mask_file_path = os.path.join(test_path, img_id, pred_mask_file_name)
            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(mask_file_path, pred_mask_file_path, logger, category_list, new_mask_label_file_name)
            else:
                logger.warn(f"not exists: {mask_file_path} or {pred_mask_file_path}")
