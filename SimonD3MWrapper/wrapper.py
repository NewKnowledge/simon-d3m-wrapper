import os
import numpy as np
import pandas as pd
import typing
import sys
from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

from Simon.penny.guesser import guess
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame, List as d3m_List
from d3m.metadata import hyperparams, base as metadata_base, params

import tensorflow as tf
import logging

__author__ = 'Distil'
__version__ = '1.2.2'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    overwrite = hyperparams.UniformBool(default = False, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='whether to overwrite manual annotations with SIMON annotations')
    statistical_classification = hyperparams.UniformBool(default = False, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='whether to append categorical / ordinal annotations using rule-based classification')
    multi_label_classification = hyperparams.UniformBool(default = True, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='whether to perfrom multi-label classification and append multiple annotations to metadata')
    max_rows = hyperparams.UniformInt(
        lower = 100, 
        upper = 2000, 
        default = 500, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'maximum number of rows to consider when classifying data type of specific column')
    max_chars = hyperparams.UniformInt(
        lower = 1, 
        upper = 100, 
        default = 20, 
        upper_inclusive=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'maximum number of characters to consider when processing row')
    p_threshold = hyperparams.Uniform(
        lower = 0, 
        upper = 1.0, 
        default = 0.5, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = """probability threshold to use when decoding classification results. 
            Predictions above p_threshold will be returned""")

class simon(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        The primitive infers the semantic type of each column from a LSTM-FCN neural network trained on 18
        different semantic types. The primitive's annotations will overwrite the default annotations if 'overwrite' 
        is set to True and will annotate columns with multiple annotations if multi_label_classification is set to 'True'.
        Finally, a different mode of categorical and ordinal classification using rule-based heuristics can be activated if
        'statistical_classification' is set to True. 
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d2fa8df2-6517-3c26-bafc-87b701c4043a",
        'version': __version__,
        'name': "simon",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Data Type Predictor','Semantic Classification','Text','NLP','Tabular'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/simon-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@{git_commit}#egg=SimonD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        },
            {
            "type": "TGZ",
            "key": "simon_models_1",
            "file_uri": "http://public.datadrivendiscovery.org/simon_models_1.tar.gz",
            "file_digest":"d071106b823ab1168879651811dd03b829ab0728ba7622785bb5d3541496c45f"
        },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_cleaning.column_type_profiler.Simon',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self.volumes = volumes
        self.X_train = None
        self.training_metadata_annotations = None

    def _produce_annotations(self, *, inputs: Inputs) -> Outputs:
        """
        Parameters
        ----------
        inputs: Input pd frame

        Returns
        -------
        Outputs
            The outputs is two lists of lists, each has length equal to number of columns in input pd frame.
            Each entry of the first one is a list of strings corresponding to each column's multi-label classification.
            Each entry of the second one is a list of floats corresponding to prediction probabilities.
        """

        frame = inputs.copy()
        checkpoint_dir = self.volumes["simon_models_1"]+"/simon_models_1/pretrained_models/"
        if self.hyperparams['statistical_classification']:
            execution_config = "Base.pkl"
            category_list = "/Categories.txt"
        else:
            execution_config = "Base_stat_geo.pkl"
            category_list = "/Categories_base_stat_geo.txt"
        with open(self.volumes["simon_models_1"] + "/simon_models_1" + category_list,'r') as f:
            Categories = f.read().splitlines()
        
        # orient the user a bit
        logger.debug("fixed categories are: ")
        Categories = sorted(Categories)
        logger.debug(Categories)
        category_count = len(Categories)

        # load specified execution configuration
        if execution_config is None:
            raise TypeError("No model config")
        Classifier = Simon(encoder={}) 
        config = Classifier.load_config(execution_config, checkpoint_dir)
        encoder = config['encoder']
        checkpoint = config['checkpoint']

        X = encoder.encodeDataFrame(frame)

        # build classifier model
        model = Classifier.generate_model(self.hyperparams['max_chars'], 
            self.hyperparams['max_rows'], 
            category_count)
        Classifier.load_weights(checkpoint, None, model, checkpoint_dir)

        model_compile = lambda m: m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['binary_accuracy'])
        model_compile(model)

        y = model.predict_on_batch(tf.constant(X))

        # discard empty column edge case
        #y[np.all(frame.isnull(),axis=0)]=0

        result = encoder.reverse_label_encode(y,self.hyperparams['p_threshold'])
        
        ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
        category_count = 0
        ordinal_count = 0
        raw_data = frame.values
        for i in np.arange(raw_data.shape[1]):
            if self.hyperparams['statistical_classification']:
                logger.info("Beginning Guessing categorical/ordinal classifications...")
                tmp = guess(raw_data[:,i], for_types ='category')
                if tmp[0]=='category':
                    category_count += 1
                    tmp2 = list(result[0][i])
                    tmp2.append('categorical')
                    result[0][i] = tmp2
                    result[1][i].append(1)
                    if ('int' in result[1][i]) or ('float' in result[1][i]) \
                        or ('datetime' in result[1][i]):
                            ordinal_count += 1
                            tmp2 = list(result[0][i])
                            tmp2.append('ordinal')
                            result[0][i] = tmp2
                            result[1][i].append(1)
                logger.info("Done with statistical variable guessing")
                ## FINISHED LABELING COMBINED DATA AS CATEGORICAL/ORDINAL
            result[0][i] = d3m_List(result[0][i])
            result[1][i] = d3m_List(result[1][i])
        Classifier.clear_session()

        out_df = pd.DataFrame.from_records(list(result)).T
        out_df.columns = ['semantic types','probabilities']
        return out_df

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Learns column annotations using training data. Saves to apply to testing data. 
        """
        # calculate SIMON annotations
        simon_annotations = self._produce_annotations(inputs = self.X_train)

        # overwrite or augment metadata with SIMON annotations
        self.training_metadata_annotations = {}
        for i in range(0, self.X_train.shape[1]):
            metadata = self.X_train.metadata.query_column(i)
            # semantic types
            if self.hyperparams['overwrite'] or 'semantic_types' not in metadata.keys():
                col_dict = dict(metadata)
                ann = simon_annotations['semantic types'][i]
                annotations_dict = {'categorical': ('https://metadata.datadrivendiscovery.org/types/CategoricalData',), 
                                    'email': ('http://schema.org/email',),
                                    'text': ('http://schema.org/Text',),
                                    'uri': ('https://metadata.datadrivendiscovery.org/types/FileName',),
                                    'address': ('http://schema.org/address',),
                                    'state': ('http://schema.org/State',),
                                    'city': ('http://schema.org/City',),
                                    'postal_code': ('http://schema.org/postalCode',),
                                    'latitude': ('http://schema.org/latitude',),
                                    'longitude': ('http://schema.org/longitude',),
                                    'country': ('http://schema.org/Country',),
                                    'country_code': ('http://schema.org/addressCountry',),
                                    'boolean': ('http://schema.org/Boolean',),
                                    'datetime': ('http://schema.org/DateTime',),
                                    'float': ('http://schema.org/Float',),
                                    'int': ('http://schema.org/Integer',),
                                    'phone': ('https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber',),
                                    'ordinal': ('https://metadata.datadrivendiscovery.org/types/OrdinalData',)}                    
                annotations = ()
                if self.hyperparams['multi_label_classification']:         
                    for key in annotations_dict:
                        if key in ann:
                            annotations = annotations + annotations_dict[key]
                else:
                    index = simon_annotations['probabilities'][i].index(max(simon_annotations['probabilities'][i]))
                    ann = ann[index]
                    for key in annotations_dict:
                        if key in ann:
                            annotations = annotations + annotations_dict[key]
                            break
                            
                # add attribute / index / target metadata to annotations tuple
                semantic_types = metadata['semantic_types'] if 'semantic_types' in metadata.keys() else ''
                # special case: if the column is the d3mIndex must have two and only two semantic types: Integer and PrimaryKey
                if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                    annotations = annotations_dict['int'] + ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
                    annotations = annotations + ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)
                if 'https://metadata.datadrivendiscovery.org/types/Attribute' in semantic_types:
                    annotations = annotations + ('https://metadata.datadrivendiscovery.org/types/Attribute',)
                if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                    annotations = annotations + ('https://metadata.datadrivendiscovery.org/types/Target',)
                if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types:
                    annotations = annotations + ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)
                col_dict['semantic_types'] = annotations
                self.training_metadata_annotations[i] = col_dict
        
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Set primitive's training data

        Parameters
        ----------
        inputs : Input pd frame

        Returns
        -------
        Outputs
            None
        """
        self.X_train = inputs

    def produce_metafeatures(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best guess for the structural type of each input column.

        Parameters
        ----------
        inputs : D3M Dataframe object

        Returns
        -------
        Outputs
            The outputs is two lists of lists, each has length equal to number of columns in input pd frame.
            Each entry of the first one is a list of strings corresponding to each column's multi-label classification.
            Each entry of the second one is a list of floats corresponding to prediction probabilities.
        """

        out_df = self._produce_annotations(inputs = inputs)

        # add metadata to output data frame
        simon_df = d3m_DataFrame(out_df)
        # first column ('semantic types')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("this is text")
        col_dict['name'] = 'semantic types'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        simon_df.metadata = simon_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('probabilities')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("this is text")
        col_dict['name'] = 'probabilities'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        simon_df.metadata = simon_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        
        return CallResult(simon_df)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Inputs]:
        """
        Add SIMON annotations if manual annotations do not exist. Hyperparameter overwrite controls whether manual 
        annotations should be overwritten with SIMON annotations.

        Parameters
        ----------
        inputs : Input pd frame

        Returns
        -------
        Outputs
            Input pd frame with metadata augmented and optionally overwritten
        """
        if len(self.training_metadata_annotations) != 0:
            for i in range(0, inputs.shape[1]):
                inputs.metadata = inputs.metadata.update_column(i, self.training_metadata_annotations[i])
        return CallResult(inputs)
