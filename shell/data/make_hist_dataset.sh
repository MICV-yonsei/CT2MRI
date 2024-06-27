for HW in 160
do
    CT_name="CT_preprocessed_$HW.nii"
    MR_name="MR_preprocessed_$HW.nii"

    for which in "train" "valid" "test"
    do
        # for plane in "axial" "sagittal" "coronal"
        for plane in "axial"
        do
            for hist_type in "normal" "avg" "colin"
            do
            python -u /root_dir/code/brain_dataset_utils/generate_total_hist_global.py \
                    --plane $plane\
                    --hist_type $hist_type \
                    --which_set $which \
                    --height $HW \
                    --width $HW \
                    --pkl_name "/root_dir/datasets/MR_hist_global_${HW}_${which}_${plane}_$hist_typ.pkl" \
                    --data_dir "/root_dir/datasets/raw_data" \
                    --data_csv "/root_dir/datasets/dataset_split.csv" \
                    --CT_name $CT_name \
                    --MR_name $MR_name \
                    > /root_dir/datasets/hdf5_log/MR_hist_global_${HW}_${which}_${plane}_$hist_type.log
            done      
        done      
    done
done      
