cd ../../../datasets
Datasets=('185_baseball_MIN_METADATA' '1491_one_hundred_plants_margin_MIN_METADATA' 'LL0_1100_popularkids_MIN_METADATA' '38_sick_MIN_METADATA' '4550_MiceProtein_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_acled_reduced_MIN_METADATA')

for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done
