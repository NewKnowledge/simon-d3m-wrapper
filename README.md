# Simon D3M Wrapper
Wrapper of the Simon semantic classifier into D3M infrastructure.  

Simon uses a LSTM-FCN neural network trained on 18 different semantic types to infer the semantic
type of each column. A hyperparameter **return_result** controls whether Simon's inferences replace existing metadata, append new columns with inferred metadata, or return a new dataframe with only the inferred columns. 

Simon can append multiple annotations if the hyperparameter **multi_label_classification** is set to 'True'. If **statistical_classification** is set to True, Simon will use rule-based heuristics to label categorical and ordinal columns. Finally, the **p_threshold** hyperparameter varies the prediction probability threshold for adding annotations. 

The following annotations will only be considered if **statistical_classification** is set to False:
    "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber"
    "http://schema.org/addressCountry"
    "http://schema.org/Country"
    "http://schema.org/longitude" 
    "http://schema.org/latitude"
    "http://schema.org/postalCode" 
    "http://schema.org/City"
    "http://schema.org/State" 
    "http://schema.org/address" 
    "http://schema.org/email" 
    "https://metadata.datadrivendiscovery.org/types/FileName"

The following annotations will only be considered if **statistical_classification** is set to True:
    "https://metadata.datadrivendiscovery.org/types/OrdinalData"

The base library for SIMON can be found here: https://github.com/NewKnowledge/simon

## Install

pip install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git#egg=SimonD3MWrapper 

