#!/bin/bash -e 

cd /

Datasets=('185_baseball' '1491_one_hundred_plants_margin' 'LL0_1100_popularkids' '38_sick' '4550_MiceProtein' '57_hypothyroid' 'LL0_acled_reduced')
Bools=('True' 'False')
cd /primitives/v2019.6.7/Distil/d3m.primitives.data_cleaning.column_type_profiler.Simon/1.2.1/pipelines
mkdir test_pipeline
mkdir best_pipelines
# create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, STAT_CLASS, MULTI_LABEL, EXECUTION TIME" >> scores.txt
cd test_pipeline
mkdir ../experiments

file="/src/simond3mwrapper/SimonD3MWrapper/simon_pipeline.py"
match="step_1.add_output('produce')"
insert="temporary line statistical_classification"
sed -i "s/$match/$match\n$insert/" $file
insert="temporary line multi_label_classification"
sed -i "s/$match/$match\n$insert/" $file

for i in "${Datasets[@]}"; do
  best_score=0
  for n in "${Bools[@]}"
    for m in "${Bools[@]}"

      # change HPs
      sed -i '/statistical_classification/d' $file
      sed -i '/multi_label_classification/d' $file
      insert="step_1.add_hyperparameter(name='statistical_classification', argument_type=ArgumentType.VALUE,data=$n)"
      sed -i "s/$match/$match\n$insert/" $file
      insert="step_1.add_hyperparameter(name='multi_label_classification', argument_type=ArgumentType.VALUE,data=$m)"
      sed -i "s/$match/$match\n$insert/" $file
      # generate and save pipeline + metafile
      python3 "/src/simond3mwrapper/SimonD3MWrapper/simon_pipeline.py" $i

      # test and score pipeline
      start=`date +%s` 
      python3 -m d3m runtime -d /datasets/ -v / fit-score -m *.meta -p *.json -c scores.csv
      end=`date +%s`
      runtime=$((end-start))

      if [ $runtime -lt 3600 ]; then
        echo "$i took less than 1 hour, evaluating pipeline"
        IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
        echo "$score"
        echo "$best_score"
        if [[ $score > $best_score ]]; then
          echo "$i, $score, $n, $m, $runtime" >> ../scores.txt
          best_score=$score
          echo "$best_score"
          cp *.meta ../experiments/
          cp *.json ../experiments/
        fi
      fi
        
      # cleanup temporary file
      rm *.meta
      rm *.json
      rm scores.csv
done
