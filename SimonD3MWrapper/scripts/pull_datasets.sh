cd /datasets
Datasets=('185_baseball' '1491_one_hundred_plants_margin' 'LL0_1100_popularkids' '38_sick' '4550_MiceProtein' '57_hypothyroid' 'LL0_acled_reduced')

for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done