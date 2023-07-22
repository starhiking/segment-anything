import os


# python tools/save_vit_encoder_results_and_all_masks.py --input dataset/FoodSeg103/Images/img_dir/test --output new_dataset/FoodSeg103/test --model-type vit_h --checkpoint ckpts/sam_vit_h_4b8939.pth 

input_root_dir = "dataset"
output_root_dir = "new_dataset"

inputs = [f"{input_root_dir}/FoodSeg103/Images/img_dir/test", f"{input_root_dir}/FoodSeg103/Images/img_dir/train", f"{input_root_dir}/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/img", f"{input_root_dir}/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/train/img"]
outputs = [f"{output_root_dir}/FoodSeg103/test", f"{output_root_dir}/FoodSeg103/train", f"{output_root_dir}/UECFOODPIXCOMPLETE/test", f"{output_root_dir}/UECFOODPIXCOMPLETE/train"]

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
