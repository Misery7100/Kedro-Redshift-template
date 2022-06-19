import keras.backend as K
from keras.layers import Layer, InputSpec
from typing import Any, List

# ------------------------- #

class ClusteringLayer(Layer):

    def __init__(

            self, 
            n_clusters: int, 
            weights: Any = None, 
            alpha: float = 1.0, 
            **kwargs

        ):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    # ......................... #

    def build(self, input_shape: List[int]):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.input_spec = InputSpec(

                dtype=K.floatx(), 
                shape=(None, input_dim)

            )
        self.clusters = self.add_weight(

                shape=(self.n_clusters, input_dim), 
                initializer='glorot_uniform', 
                name='clusters'

            )

        if self.initial_weights:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    # ......................... #

    def call(self, inputs: Any, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))

        return q

    # ......................... #

    def compute_output_shape(self, input_shape: List[int]):

        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    # ......................... #

    def get_config(self):

        config = {'n_clusters': self.n_clusters}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

# ------------------------- #