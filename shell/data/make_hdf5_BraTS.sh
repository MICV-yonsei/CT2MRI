
src_name="t2f"
tgt_name="t1n"

root_dir=/root_dir
for which in "train" "valid" "test"
do
    # for plane in "axial" "sagittal" "coronal"
    for plane in "axial"
    do
    python -u brain_dataset_utils/generate_total_hdf5_csv_BraTS.py \
            --plane  $plane\
            --root_dir $which \
            --which_set $root_dir \
            --hdf5_name "$root_dir/datasets/BraTS_${src_name}_to_${tgt_name}_${which}_${plane}.hdf5" \
            --src_name $src_name \
            --tgt_name $tgt_name \
            > $root_dir/datasets/hdf5_log/BraTS_${src_name}_to_${tgt_name}_${which}_${plane}.log
    done      
done
