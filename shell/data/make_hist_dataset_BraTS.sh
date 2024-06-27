
src_name="t2f"
tgt_name="t1n"

root_dir=/root_dir
for which in "train" "valid" "test"
do
    # for plane in "axial" "sagittal" "coronal"
    for plane in "axial"
    do
        for hist_type in "normal" "avg" "colin"
        do
        python -u $root_dir/code/brain_dataset_utils/generate_total_hist_global_BraTS.py \
                --plane  $plane\
                --hist_type $hist_type \
                --which_set $which \
                --root_dir $root_dir \
                --pkl_name "$root_dir/datasets/MR_hist_global_${tgt_name}_${which}_${plane}_BraTS_$hist_typ.pkl" \
                --src_name $src_name \
                --tgt_name $tgt_name \
                > $root_dir/datasets/hdf5_log/MR_hist_global_${tgt_name}_${which}_${plane}_BraTS_$hist_typ.log
        done
    done
done

