#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createFeatureMap.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="createFeatureMap.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly MODEL_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly MODEL_NAME=$(cat ${JSON_FILE} | jq -r ".model_name")
readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly IMAGE_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".image_patch_width")
readonly LABEL_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".label_patch_width")
readonly PLANE_SIZE=$(cat ${JSON_FILE} | jq -r ".plane_size")

readonly NUM_ARRAYS=$(cat ${JSON_FILE} | jq -r ".num_arrays")
readonly KEYS=$(cat ${JSON_FILE} | jq -r ".num_arrays | keys[]")

for key in ${KEYS[@]}
do
 echo "----------$key----------"
 num_array=$(echo $NUM_ARRAYS | jq -r ".$key[]")
 echo $num_array
 for number in ${num_array[@]}
 do
     image_path="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
     save_path="${SAVE_DIRECTORY}/${key}/case_${number}"
     model_path="${MODEL_DIRECTORY}/${key}/${MODEL_NAME}"

     echo "image_path:${image_path}"
     echo "model_path:${model_path}"
     echo "SAVE_PATH:${save_path}"
     echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
     echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
     echo "IMAGE_PATCH_WIDTH:${IMAGE_PATCH_WIDTH}"
     echo "LABEL_PATCH_WIDTH:${LABEL_PATCH_WIDTH}"
     echo "PLANE_SIZE:${PLANE_SIZE}"

     if [ $MASK_NAME = "No" ]; then
      mask=""

     else
      mask_path="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"
      echo "MASK_PATH:${mask_path}"
      mask="--mask_path ${mask_path}"

     fi

     python3 createFeatureMap.py ${image_path} ${model_path} ${save_path} --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --image_patch_width ${IMAGE_PATCH_WIDTH} --label_patch_width ${LABEL_PATCH_WIDTH} --plane_size ${PLANE_SIZE} ${mask}

     # Judge if it works.
     if [ $? -eq 0 ]; then
      echo "Done."

     else
      echo "Fail"

     fi
  done
 done
