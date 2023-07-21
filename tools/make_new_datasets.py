import os


# python tools/save_vit_encoder_results_and_all_masks.py --input dataset/FoodSeg103/Images/img_dir/test --output new_dataset/FoodSeg103/test --model-type vit_h --checkpoint ckpts/sam_vit_h_4b8939.pth 

inputs = ["dataset/FoodSeg103/Images/img_dir/test", "dataset/FoodSeg103/Images/img_dir/train", "dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/img", "dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/train/img"]
outputs = ["new_dataset/FoodSeg103/test", "new_dataset/FoodSeg103/train", "new_dataset/UECFOODPIXCOMPLETE/test", "new_dataset/UECFOODPIXCOMPLETE/train"]

CUDA_IDs = [0,1]

if __name__ == "__main__":

    cmd = ""
    for i in range(len(CUDA_IDs)):
        CUDA_ID = CUDA_IDs[i]

        part_inputs = inputs[2*i : 2*(i+1)]
        part_outputs = outputs[2*i : 2*(i+1)]

        for input, output in zip(part_inputs, part_outputs):
            cmd += f"CUDA_VISIBLE_DEVICES={CUDA_ID} python tools/save_vit_encoder_results_and_all_masks.py --input {input} --output {output} --points-per-batch 24 --model-type vit_h --checkpoint ckpts/sam_vit_h_4b8939.pth && "

        cmd = cmd[:-3] + " & "
    
    # cmd = cmd[:-3]
    print(cmd)
    os.system(cmd)
