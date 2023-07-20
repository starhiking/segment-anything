import os
import cv2
import logging

food103_img_paths = ["dataset/FoodSeg103/Images/img_dir/train", "dataset/FoodSeg103/Images/img_dir/test"]
food103_mask_paths = ["dataset/FoodSeg103/Images/ann_dir/train", "dataset/FoodSeg103/Images/ann_dir/test"]

UECFoodPix_img_paths = ["dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/train/img", "dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/img"]
UECFoodPix_mask_paths = ["dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/train/mask", "dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/mask"]

# NOTE: img is jpg and mask is png

def create_logger():
    
    log_file = f"foodseg_wh_warning.log"
    final_log_file = os.path.join("tools", log_file)

    # if os.path.exists(final_log_file):
    #     print("Current log file is exist")
    #     raise ValueError("Log file alread exist")

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

def check_wh_correct(img_folder, mask_folder, logger):
    img_paths = os.listdir(img_folder)
    mask_paths = [img_path[:-4]+".png" for img_path in img_paths]
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(os.path.join(img_folder, img_path))
        mask = cv2.imread(os.path.join(mask_folder, mask_path))

        if img is None or mask is None:
            logger.error(f"Image {os.path.join(img_folder, img_path)} or mask {os.path.join(mask_folder, mask_path)} is None")
        elif img.shape[:2] != mask.shape[:2]:
            logger.warning(f"Image {os.path.join(img_folder, img_path)} shape is {img.shape}. And mask {os.path.join(mask_folder, mask_path)} shape is {mask.shape}")

if __name__ == '__main__':
    logger = create_logger()
    
    # UECFoodPix
    for img_folder, mask_folder in zip(UECFoodPix_img_paths, UECFoodPix_mask_paths):
        logger.info(f"Check {img_folder} and {mask_folder}")
        check_wh_correct(img_folder, mask_folder, logger)
    
    # food103
    for img_folder, mask_folder in zip(food103_img_paths, food103_mask_paths):
        logger.info(f"Check {img_folder} and {mask_folder}")
        check_wh_correct(img_folder, mask_folder, logger)
    
    print("Done") 
