class Features:
    def __init__(self, numeric_features=None, categorical_features=None, pipeline=None):

        self._numeric_features = numeric_features

        self._categorical_features = self._detect_cat_features(pipeline, categorical_features)

    @staticmethod
    def _detect_cat_features(pipeline, categorical_features):
        return pipeline.named_steps['transformers_pipeline'].named_transformers_["cat"] \
            .named_steps['one_hot_encoding'].get_feature_names(categorical_features)

    @property
    def all(self):
        return list(self._numeric_features) + list(self._categorical_features)

    @property
    def numerical(self):
        return self._numeric_features

    @property
    def nb_numerical(self):
        return len(self._numeric_features)

    @property
    def categorical(self):
        return self._categorical_features

    @property
    def nb_categorical(self):
        return len(self._categorical_features)
