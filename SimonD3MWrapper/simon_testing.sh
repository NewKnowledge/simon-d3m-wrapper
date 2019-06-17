#!/bin/bash -e 

cd /
#git clone https://gitlab.com/jgleason/primitives
cd /primitives
#git branch simon_pipelines
git checkout simon_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

#Datasets=('185_baseball')
Datasets=('LL0_acled_reduced')
# '1491_one_hundred_plants_margin' 'LL0_1100_popularkids' 'LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon')
rm  /primitives/v2019.6.7/Distil/d3m.primitives.data_cleaning.column_type_profiler.Simon/1.2.1/pipelines/test_pipeline/*
cd /primitives/v2019.6.7/Distil/d3m.primitives.data_cleaning.column_type_profiler.Simon/1.2.1/pipelines
#mkdir test_pipeline
cd test_pipeline

# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/simond3mwrapper/SimonD3MWrapper/simon_pipeline.py" $i

  # test and score pipeline
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ -v / fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
     echo "$i took less than 1 hour, copying pipeline"
     #cp * ../
  fi

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  #echo "$i, $score, $runtime" >> scores.txt
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done
