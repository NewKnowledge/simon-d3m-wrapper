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
from d3m.primitive_interfaces.unsupervised_learning import (
    UnsupervisedLearnerPrimitiveBase,
)
from d3m.primitive_interfaces.base import CallResult
from d3m.exceptions import PrimitiveNotFittedError

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame, List as d3m_List
from d3m.metadata import hyperparams, base as metadata_base, params

import tensorflow as tf
import logging

__author__ = "Distil"
__version__ = "1.2.2"
__contact__ = "mailto:jeffrey.gleason@yonder.co"

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

COLUMN_TYPES_DICT = {
    "categorical": ("https://metadata.datadrivendiscovery.org/types/CategoricalData",),
    "email": ("http://schema.org/email",),
    "text": ("http://schema.org/Text",),
    "uri": ("https://metadata.datadrivendiscovery.org/types/FileName",),
    "address": ("http://schema.org/address",),
    "state": ("http://schema.org/State",),
    "city": ("http://schema.org/City",),
    "postal_code": ("http://schema.org/postalCode",),
    "latitude": ("http://schema.org/latitude",),
    "longitude": ("http://schema.org/longitude",),
    "country": ("http://schema.org/Country",),
    "country_code": ("http://schema.org/addressCountry",),
    "boolean": ("http://schema.org/Boolean",),
    "datetime": (
        "http://schema.org/DateTime",
        "https://metadata.datadrivendiscovery.org/types/Time",
    ),
    "float": ("http://schema.org/Float",),
    "int": ("http://schema.org/Integer",),
    "phone": ("https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber",),
    "ordinal": ("https://metadata.datadrivendiscovery.org/types/OrdinalData",),
}

COLUMN_ROLES = [
    "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
    "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
    "https://metadata.datadrivendiscovery.org/types/UniqueKey",
    "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
    "https://metadata.datadrivendiscovery.org/types/Target",
    "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    "https://metadata.datadrivendiscovery.org/types/Attribute",
    "https://metadata.datadrivendiscovery.org/types/GroupingKey",
    "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",
]


class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    overwrite = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to overwrite manual annotations with SIMON annotations",
    )
    statistical_classification = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="whether to append categorical / ordinal annotations using rule-based classification",
    )
    multi_label_classification = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="whether to perfrom multi-label classification and append multiple annotations to metadata",
    )
    max_rows = hyperparams.UniformInt(
        lower=100,
        upper=2000,
        default=500,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="maximum number of rows to consider when classifying data type of specific column",
    )
    max_chars = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=20,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="maximum number of characters to consider when processing row",
    )
    p_threshold = hyperparams.Uniform(
        lower=0,
        upper=1.0,
        default=0.5,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="""probability threshold to use when decoding classification results. 
            Predictions above p_threshold will be returned""",
    )


class simon(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """ The primitive uses a LSTM-FCN neural network trained on 18 different semantic types to infer the semantic
        type of each column. The primitive's annotations will overwrite the default annotations if 'overwrite'
        is set to True (column roles, e.g. Attribute, PrimaryKey, Target from original annotations will be kept).
        Otherwise the primitive will augment the existing annotations with its predicted labels.
        The primitive will append multiple annotations if multi_label_classification is set to 'True'.
        Finally, a different mode of typing inference that uses rule-based heuristics will be used  if
        'statistical_classification' is set to True.

        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object

        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
            volumes {Dict[str, str]} -- large file dictionary containing model weights (default: {None})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "d2fa8df2-6517-3c26-bafc-87b701c4043a",
            "version": __version__,
            "name": "simon",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": [
                "Data Type Predictor",
                "Semantic Classification",
                "Text",
                "NLP",
                "Tabular",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    # Unstructured URIs.
                    "https://github.com/NewKnowledge/simon-d3m-wrapper",
                ],
            },
            # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
            # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
            # install a Python package first to be even able to run setup.py of another package. Or you have
            # a dependency which is not on PyPi.
            "installation": [
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@{git_commit}#egg=SimonD3MWrapper".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
                {
                    "type": "TGZ",
                    "key": "simon_models_1",
                    "file_uri": "http://public.datadrivendiscovery.org/simon_models_1.tar.gz",
                    "file_digest": "d071106b823ab1168879651811dd03b829ab0728ba7622785bb5d3541496c45f",
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.data_cleaning.column_type_profiler.Simon",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_CLEANING,
        }
    )

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: typing.Dict[str, str] = None,
    ) -> None:

        super().__init__(
            hyperparams=hyperparams, random_seed=random_seed, volumes=volumes
        )
        self.volumes = volumes
        self.X_train = None
        self.training_metadata_annotations = None
        self.params = None

    def _produce_annotations(self, *, inputs: Inputs) -> Outputs:
        """ generates dataframe with semantic type classifications and classification probabilities
            for each column of original dataframe

        Arguments:
            inputs {Inputs} -- D3M dataframe

        Returns:
            Outputs -- dataframe with two columns: "semantic type classifications" and "probabilities"
                       Each row represents a column in the original dataframe. The column "semantic type
                       classifications" contains a list of all semantic type labels and the column
                       "probabilities" contains a list of the model's confidence in assigning each
                       respective semantic type label
        """

        # load model checkpoint
        checkpoint_dir = (
            self.volumes["simon_models_1"] + "/simon_models_1/pretrained_models/"
        )
        if self.hyperparams["statistical_classification"]:
            execution_config = "Base.pkl"
            category_list = "/Categories.txt"
        else:
            execution_config = "Base_stat_geo.pkl"
            category_list = "/Categories_base_stat_geo.txt"
        with open(
            self.volumes["simon_models_1"] + "/simon_models_1" + category_list, "r"
        ) as f:
            Categories = f.read().splitlines()

        # create model object
        Classifier = Simon(encoder={})
        config = Classifier.load_config(execution_config, checkpoint_dir)
        encoder = config["encoder"]
        checkpoint = config["checkpoint"]
        model = Classifier.generate_model(
            self.hyperparams["max_chars"], self.hyperparams["max_rows"], len(Categories)
        )
        Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
        )

        # prepare data and make predictions
        frame = inputs.copy()
        prepped_data = encoder.encodeDataFrame(frame)
        preds = model.predict_on_batch(tf.constant(prepped_data))
        decoded_preds = encoder.reverse_label_encode(
            preds, self.hyperparams["p_threshold"]
        )

        # apply statistical / ordinal classification if desired
        if self.hyperparams["statistical_classification"]:
            logger.debug("Beginning Guessing categorical/ordinal classifications...")
            raw_data = frame.values
            guesses = [
                guess(raw_data[:, i], for_types="category")
                for i in np.arange(raw_data.shape[1])
            ]
            for i, g in enumerate(guesses):
                if g[0] == "category":
                    decoded_preds[0][i] += ("categorical",)
                    decoded_preds[1][i].append(1)
                    if (
                        ("int" in decoded_preds[1][i])
                        or ("float" in decoded_preds[1][i])
                        or ("datetime" in decoded_preds[1][i])
                    ):
                        decoded_preds[0][i] += ("ordinal",)
                        decoded_preds[1][i].append(1)
            logger.debug("Done with statistical variable guessing")

        # clear tf session
        Classifier.clear_session()

        out_df = pd.DataFrame.from_records(list(decoded_preds)).T
        out_df.columns = ["semantic types", "probabilities"]
        return out_df

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Learns column annotations using training data. Saves to apply to testing data.

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Returns:
                CallResult[None]
        """

        # calculate SIMON annotations
        simon_annotations = self._produce_annotations(inputs=self.X_train)
        logger.debug(f"simon annotations: {simon_annotations}")

        self.training_metadata_annotations = {}
        for col_idx in range(0, self.X_train.shape[1]):

            # are we overwriting or augmenting
            metadata = self.X_train.metadata.query_column(col_idx)
            metadata_dict = dict(metadata)
            if "semantic_types" not in metadata.keys():
                original_semantic_types = ""
                new_annotations = ()
            elif self.hyperparams["overwrite"]:
                original_semantic_types = metadata["semantic_types"]
                new_annotations = ()
            else:
                original_semantic_types = metadata["semantic_types"]
                new_annotations = original_semantic_types

            # append simon labels
            simon_labels = simon_annotations["semantic types"][col_idx]
            if self.hyperparams["multi_label_classification"]:
                for label in simon_labels:
                    new_annotations += COLUMN_TYPES_DICT[label]
            else:
                index = simon_annotations["probabilities"][i].index(
                    max(simon_annotations["probabilities"][i])
                )
                new_annotations += COLUMN_TYPES_DICT[simon_labels[index]]

            # add column role information to annotations tuple
            for col_type in original_semantic_types:
                # if the column is PrimaryKey add Integer annotation and delete other annotations
                if col_type == COLUMN_ROLES[0]:
                    new_annotations = COLUMN_TYPES_DICT["int"]
                if col_type in COLUMN_ROLES:
                    new_annotations += (col_type,)
            metadata_dict["semantic_types"] = new_annotations
            self.training_metadata_annotations[col_idx] = metadata_dict

        self._is_fit = True
        return CallResult(None)

    def get_params(self) -> Params:
        return self.params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs) -> None:
        """ Sets primitive's training data

            Arguments:
                inputs {Inputs} -- D3M dataframe
        """
        self.X_train = inputs

    def produce_metafeatures(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's best guess for the structural type of each input column.

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit

            Returns:
                CallResult[Outputs] -- dataframe with two columns: "semantic type classifications" and "probabilities"
                    Each row represents a column in the original dataframe. The column "semantic type
                    classifications" contains a list of all semantic type labels and the column
                    "probabilities" contains a list of the model's confidence in assigning each
                    respective semantic type label
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        out_df = self._produce_annotations(inputs=inputs)

        # add metadata to output data frame
        simon_df = d3m_DataFrame(out_df)
        # first column ('semantic types')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = typing.List[str]
        col_dict["name"] = "semantic types"
        col_dict["semantic_types"] = (
            "http://schema.org/Text",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )
        simon_df.metadata = simon_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )
        # second column ('probabilities')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict["structural_type"] = typing.List[float]
        col_dict["name"] = "probabilities"
        col_dict["semantic_types"] = (
            "http://schema.org/Text",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )
        simon_df.metadata = simon_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 1), col_dict
        )

        return CallResult(simon_df, has_finished=self._is_fit)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Inputs]:
        """ Add SIMON annotations if manual annotations do not exist. Hyperparameter overwrite controls
            whether SIMON annotations should overwrite manual annotations or merely augment them

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit

            Returns:
                CallResult[Outputs] -- Input pd frame with metadata augmented and optionally overwritten

        """
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        if len(self.training_metadata_annotations) != 0:
            for i in range(0, inputs.shape[1]):
                if (
                    len(
                        {
                            "https://metadata.datadrivendiscovery.org/types/Target",
                            "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                            "https://metadata.datadrivendiscovery.org/types/Attribute",
                        }
                        & set(self.training_metadata_annotations[i]["semantic_types"])
                    )
                    == 0
                ):
                    self.training_metadata_annotations[i][
                        "semantic_types"
                    ] = self.training_metadata_annotations[i]["semantic_types"] + (
                        "https://metadata.datadrivendiscovery.org/types/Attribute",
                    )

                inputs.metadata = inputs.metadata.update_column(
                    i, self.training_metadata_annotations[i]
                )
        inputs.metadata = inputs.metadata.update(
            (metadata_base.ALL_ELEMENTS,),
            {"dimension": {"length": len(self.training_metadata_annotations),},},
        )

        return CallResult(inputs, has_finished=self._is_fit)
