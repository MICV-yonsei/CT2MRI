
for HW in 160 
do
    CT_name="CT_preprocessed_$HW.nii"
    MR_name="MR_preprocessed_$HW.nii"

    for which in "train" "valid" "test"
    do
        # for plane in "axial" "sagittal" "coronal"
        for plane in "axial"
        do
        python -u brain_dataset_utils/generate_total_hdf5_csv.py \
                --plane  $plane\
                --which_set $which \
                --height $HW \
                --width $HW \
                --hdf5_name "/root_dir/datasets/${HW}_${which}_${plane}.hdf5" \
                --data_dir "/root_dir/datasets/raw_data" \
                --data_csv "/root_dir/datasets/dataset_split.csv" \
                --CT_name $CT_name \
                --MR_name $MR_name \
                > /root_dir/datasets/hdf5_log/${HW}_${which}_${plane}.log
        done      
    done
done      
