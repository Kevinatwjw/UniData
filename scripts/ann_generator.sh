PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 1. 修改 data_path
# 注意：这里必须写到 'nuscenes' 这一层，不要写到 'v1.0-mini'
# 必须使用绝对路径（以 / 开头）以避免找不到文件
# data_path="/root/autodl-tmp/UniScene_Storage/data/nuscenes"
data_path="./data/nuscenes"

# 2. 修改 data_version
# 将默认的 trainval 改为 mini
data_version="v1.0-mini"
# data_version="v1.0-trainval"

ann_frequency=$1
PY_ARGS=${@:2}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/$input_frequency/'2'_$ann_frequency
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_annotation_generator \
    --data_path $data_path \
    --data_version $data_version \
    --ann_frequency $ann_frequency \
    $PY_ARGS | tee -a $LOG_DIR/log_generate_ann.txt
