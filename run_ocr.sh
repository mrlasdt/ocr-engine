#bash scripts/run_ocr.sh -i /mnt/hdd2T/hoanglv/Datasets/FWD -o results/ocr/FWD/230306_forms -e out.csv -k "{\"device\":\"cuda:1\"}" -x True
#bash scripts/run_ocr.sh -i '/mnt/hdd2T/hoanglv/Datasets/FWD/01_Classified_forms/BC\ kiem\ tra\ y\ te/' -o results/ocr/FWD/230306_forms/01_Classified_forms -e out.csv -k "{\"device\":\"cuda:1\"}" -x 1
export PYTHONWARNINGS="ignore"

img=""
out_dir=""
base_dir=""
export_csv="out.csv"
export_img="True"
ocr_kwargs="{\"device\":\"cuda:1\"}"

while getopts i:o:b:e:x:k: flag
do
    case "${flag}" in
        i) img=${OPTARG};;
        o) out_dir=${OPTARG};;
        b) base_dir=${OPTARG};;
        e) export_csv=${OPTARG};;
        x) export_img=${OPTARG};;
        k) ocr_kwargs=${OPTARG};;
    esac
done

echo "python run.py --image=\"$img\" --save_dir \"$out_dir\" --base_dir \"$base_dir\" --export_csv \"$export_csv\" --export_img \"$export_img\" --ocr_kwargs \"$ocr_kwargs\""

python run.py \
    --image="$img" \
    --save_dir="$out_dir" \
    --base_dir="$base_dir" \
    --export_csv="$export_csv" \
    --export_img="$export_img" \
    --ocr_kwargs="$ocr_kwargs"