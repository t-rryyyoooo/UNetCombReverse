#!/bin/bash

# input 
readonly input_directory="input"
echo -n "is json file name pipliner.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "is json file name the same as this file name?[y/n]:"
 read which
done

# specify json file path.
if [ $which = "y" ];then
 json_name="pipliner.json"
else
 echo -n "json_file_name="
 read json_name
fi

readonly json_file="${input_directory}/${json_name}"

# run or not
readonly run_training=$(cat ${json_file} | jq -r ".run_training")
readonly run_segmentation=$(cat ${json_file} | jq -r ".run_segmentation")
readonly run_caluculation=$(cat ${json_file} | jq -r ".run_caluculation")

# training input
readonly image_path_n=$(eval echo $(cat ${json_file} | jq -r ".image_path_n"))
readonly image_path_f=$(eval echo $(cat ${json_file} | jq -r ".image_path_f"))
readonly label_path=$(eval echo $(cat ${json_file} | jq -r ".label_path"))
readonly model_savepath=$(eval echo $(cat ${json_file} | jq -r ".model_savepath"))
readonly org_model=$(eval echo $(cat ${json_file} | jq -r ".org_model"))
readonly log=$(eval echo $(cat ${json_file} | jq -r ".log"))
readonly in_channel_main=$(cat ${json_file} | jq -r ".in_channel_main")
readonly in_channel_final=$(cat ${json_file} | jq -r ".in_channel_final")
readonly num_class=$(cat ${json_file} | jq -r ".num_class")
readonly learning_rate=$(cat ${json_file} | jq -r ".learning_rate")
readonly batch_size=$(cat ${json_file} | jq -r ".batch_size")
readonly dropout=$(cat ${json_file} | jq -r ".dropout")
readonly num_workers=$(cat ${json_file} | jq -r ".num_workers")
readonly epoch=$(cat ${json_file} | jq -r ".epoch")
readonly gpu_ids=$(cat ${json_file} | jq -r ".gpu_ids")
readonly api_key=$(cat ${json_file} | jq -r ".api_key")
readonly project_name=$(cat ${json_file} | jq -r ".project_name")
readonly experiment_name=$(cat ${json_file} | jq -r ".experiment_name")
readonly org_model_name=$(cat ${json_file} | jq -r ".org_model_name")

# Segmentation input
save_directory="${image_path_f}/segmentation"
readonly data_directory=$(eval echo $(cat ${json_file} | jq -r ".data_directory"))
readonly model_fmc_dir=$(eval echo $(cat ${json_file} | jq -r ".model_fmc_dir"))
readonly image_patch_size=$(cat ${json_file} | jq -r ".image_patch_size")
readonly label_patch_size=$(cat ${json_file} | jq -r ".label_patch_size")
readonly image_patch_size_fmc=$(cat ${json_file} | jq -r ".image_patch_size_fmc")
readonly label_patch_size_fmc=$(cat ${json_file} | jq -r ".label_patch_size_fmc")
readonly image_patch_width_fmc=$(cat ${json_file} | jq -r ".image_patch_width_fmc")
readonly label_patch_width_fmc=$(cat ${json_file} | jq -r ".label_patch_width_fmc")
readonly plane_size=$(cat ${json_file} | jq -r ".plane_size")
readonly image_name=$(cat ${json_file} | jq -r ".image_name")
readonly save_name=$(cat ${json_file} | jq -r ".save_name")
readonly model_name=$(cat ${json_file} | jq -r ".model_name")
readonly model_fmc_name=$(cat ${json_file} | jq -r ".model_fmc_name")

# Caluculate DICE input
readonly csv_savedir=$(eval echo $(cat ${json_file} | jq -r ".csv_savedir"))
readonly class_label=$(cat ${json_file} | jq -r ".class_label")
readonly true_name=$(cat ${json_file} | jq -r ".true_name")

readonly train_lists=$(cat ${json_file} | jq -r ".train_lists")
readonly val_lists=$(cat ${json_file} | jq -r ".val_lists")
readonly test_lists=$(cat ${json_file} | jq -r ".test_lists")
readonly keys=$(cat ${json_file} | jq -r ".train_lists | keys[]")

all_patients=""
for key in ${keys[@]}
do 
 echo $key
 train_list=$(echo $train_lists | jq -r ".$key")
 val_list=$(echo $val_lists | jq -r ".$key")
 test_list=$(echo $test_lists | jq -r ".$key")
 test_list=(${test_list// / })
 image_path_ff="${image_path_f}/${key}"
 model_save="${model_savepath}/${key}"
 org_model_f="${org_model}/${key}/${org_model_name}"
 l="${log}/${key}"
 ex_name="${experiment_name}_${key}"

 run_training_fold=$(echo $run_training | jq -r ".$key")
 run_segmentation_fold=$(echo $run_segmentation | jq -r ".$key")
 run_caluculation_fold=$(echo $run_caluculation | jq -r ".$key")

 if ${run_training_fold};then
  echo "---------- training ----------"
  echo "image_path_list:${image_path_n} ${image_path_ff}"
  echo "label_path:${label_path}"
  echo "model_save:${model_save}"
  echo "org_model_f:${org_model_f}"
  echo "train_list:${train_list}"
  echo "val_list:${val_list}"
  echo "log:${l}"
  echo "in_channel_main:${in_channel_main}"
  echo "in_channel_final:${in_channel_final}"
  echo "num_class:${num_class}"
  echo "learning_rate:${learning_rate}"
  echo "batch_size:${batch_size}"
  echo "num_workers:${num_workers}"
  echo "epoch:${epoch}"
  echo "dropout:${dropout}"
  echo "gpu_ids:${gpu_ids}"
  echo "api_key:${api_key}"
  echo "project_name:${project_name}"
  echo "ex_name:${ex_name}"

  python3 train.py ${image_path_n} ${image_path_ff} ${label_path} ${model_save} --org_model ${org_model_f} --train_list ${train_list} --val_list ${val_list} --log ${l} --in_channel_main ${in_channel_main} --in_channel_final ${in_channel_final} --num_class ${num_class} --learning_rate ${learning_rate} --batch_size ${batch_size} --dropout ${dropout} --num_workers ${num_workers} --epoch ${epoch} --gpu_ids ${gpu_ids} --api_key ${api_key} --project_name ${project_name} --experiment_name ${ex_name}

  if [ $? -ne 0 ];then
   exit 1
  fi

 else
  echo "---------- no training. ----------"
 fi

 model="${model_save}/${model_name}"
 mn=${model%.*}
 csv_name=${mn////_}
 if ${run_segmentation_fold};then
  echo "---------- segmentation ----------"
  echo ${test_list[@]}
  for number in ${test_list[@]}
  do
   image="${data_directory}/case_${number}/${image_name}"
   save="${save_directory}/case_${number}/${save_name}"
   model_fmc="${model_fmc_dir}/${key}/${model_fmc_name}"

   echo "image:${image}"
   echo "model_fmc:${model_fmc}"
   echo "model:${model}"
   echo "save:${save}"
   echo "image_patch_size:${image_patch_size}"
   echo "label_patch_size:${label_patch_size}"
   echo "image_patch_size:${image_patch_size_fmc}"
   echo "label_patch_size:${label_patch_size_fmc}"
   echo "image_patch_width_fmc:${image_patch_width_fmc}"
   echo "label_patch_width_fmc:${label_patch_width_fmc}"
   echo "plane_size:${plane_size}"
   echo "gpu_ids:${gpu_ids}"

   #python3 segmentation.py ${image} ${model_fmc} ${model} ${save} --image_patch_size ${image_patch_size} --label_patch_size ${label_patch_size} --image_patch_size_fmc ${image_patch_size_fmc} --label_patch_size_fmc ${label_patch_size_fmc} --plane_size ${plane_size} -g ${gpu_ids} --image_patch_width_fmc ${image_patch_width_fmc} --label_patch_width ${label_patch_width_fmc}


   if [ $? -ne 0 ];then
    exit 1
   fi

   if ${run_caluculation_fold};then
    all_patients="${all_patients}${number} "
   fi

  done
  
 else
  echo "---------- no segmentation. ----------"
 fi

done

echo "---------- caluculation ----------"
csv_savepath="${csv_savedir}/${csv_name}.csv"
echo "true_directory:${data_directory}"
echo "predict_directory:${save_directory}"
echo "csv_savepath:${csv_savepath}"
echo "all_patients:${all_patients[@]}"
echo "num_class:${num_class}"
echo "class_label:${class_label}"
echo "true_name:${true_name}"
echo "predict_name:${save_name}"


python3 caluculateDICE.py ${data_directory} ${save_directory} ${csv_savepath} ${all_patients} --classes ${num_class} --class_label ${class_label} --true_name ${true_name} --predict_name ${save_name} 

if [ $? -ne 0 ];then
 exit 1
fi

echo "---------- logging ----------"
python3 logger.py ${json_file}
echo done.


