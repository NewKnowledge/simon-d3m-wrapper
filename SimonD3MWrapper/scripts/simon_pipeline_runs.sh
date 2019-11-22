#!/bin/bash -e 

Datasets=('185_baseball' '1491_one_hundred_plants_margin' 'LL0_1100_popularkids' '38_sick' '4550_MiceProtein' '57_hypothyroid' 'LL0_acled_reduced')
cd /primitives/v2019.11.10/Distil/d3m.primitives.data_cleaning.column_type_profiler.Simon/1.2.2
cd pipelines
python3 "/src/simond3mwrapper/SimonD3MWrapper/simon_pipeline.py"
cd ..
cd pipeline_runs

for i in "${Datasets[@]}"; do

  # generate pipeline run
  python3 -m d3m runtime -v / -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O $i.yml

done

# zip pipeline runs individually
# cd ..
# gzip -r pipeline_runs