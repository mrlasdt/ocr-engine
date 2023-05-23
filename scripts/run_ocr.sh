#bash scripts/run_ocr.sh -i /mnt/ssd1T/hungbnt/DocumentClassification/data/OCR040_043 -o /mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/OCR040_043 -e out.csv -k "{\"device\":\"cuda:1\"}" -x True
export PYTHONWARNINGS="ignore"

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
echo "run.py --image=\"$img\" --save_dir \"$out_dir\" --base_dir \"$base_dir\" --export_csv \"$export_csv\" --export_img \"$export_img\" --ocr_kwargs \"$ocr_kwargs\""

python run.py \
    --image="$img" \
    --save_dir  $out_dir \
    --export_csv $export_csv\
    --export_img $export_img\
    --ocr_kwargs $ocr_kwargs\

