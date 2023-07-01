#bash scripts/run_ocr.sh -i /mnt/hdd2T/AICR/Projects/2023/FWD/Forms/PDFs/ -o /mnt/ssd1T/hungbnt/DocumentClassification/results/ocr -e out.csv -k "{\"device\":\"cuda:1\"}" -p True -n Passport 'So\ HK'
#bash scripts/run_ocr.sh -i '/mnt/hdd2T/AICR/Projects/2023/FWD/Forms/PDFs/So\ HK' -o /mnt/ssd1T/hungbnt/DocumentClassification/results/ocr -e out.csv -k "{\"device\":\"cuda:1\"}" -p True
#-n and -x do not accept multiple argument currently

export PYTHONWARNINGS="ignore"

while getopts i:o:b:e:p:k:n:x: flag
do
    case "${flag}" in
        i) img=${OPTARG};;
        o) out_dir=${OPTARG};;
        b) base_dir=${OPTARG};;
        e) export_csv=${OPTARG};;
        p) export_img=${OPTARG};;
        k) ocr_kwargs=${OPTARG};;
        n) include=("${OPTARG[@]}");;
        x) exclude=("${OPTARG[@]}");;
    esac
done

cmd="python run.py \
    --image $img \
    --save_dir $out_dir \
    --export_csv $export_csv \
    --export_img $export_img \
    --ocr_kwargs $ocr_kwargs"

if [ ${#include[@]} -gt 0 ]; then
    cmd+=" --include"
    for item in "${include[@]}"; do
        cmd+=" $item"
    done
fi

if [ ${#exclude[@]} -gt 0 ]; then
    cmd+=" --exclude"
    for item in "${exclude[@]}"; do
        cmd+=" $item"
    done
fi


echo $cmd
exec $cmd
