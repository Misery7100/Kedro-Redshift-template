import numpy as np
import tensorflow as tf
import logging

from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2 as l2_reg
from sklearn.cluster import KMeans
from typing import Any, Dict, Tuple, Union, Callable
from tqdm.auto import tqdm

from .layers import ClusteringLayer
from .callbacks import KerasTqdmBar

log = logging.getLogger(__name__)
bar_format = '{l_bar}{bar:40}{r_bar}{bar:-40b}'

# ------------------------- #

class AutoEncoder(Model):

    def __init__(

            self,
            data_shape: Tuple[int],
            latent_space_size: int = 20,
            layers_half: Tuple[int] = (100, 80, 50),
            enc_activation: Union[str, Callable] = 'relu',
            dec_activation: Union[str, Callable] = 'relu',
            last_activation: Union[str, Callable] = 'selu',
            **kwargs

        ):

        super().__init__(**kwargs)

        self.latent_space_size = latent_space_size

        init = tf.keras.initializers.VarianceScaling(

            scale=1. / 3., 
            mode='fan_in', 
            distribution='uniform'

        )

        layers_half = (data_shape[1],) + layers_half + (latent_space_size,)

        # encoder

        nst = len(layers_half) - 1
        self.ae_encoder_layers = []

        for i in range(nst - 1):
            x = Dense(
                    layers_half[i + 1], 
                    activation=enc_activation, 
                    kernel_initializer=init, 
                    name=f'encoder_{i}'
                )
            self.ae_encoder_layers.append(x)

        # latent space
        
        self.ae_latent = Dense(
                layers_half[-1],
                activation=enc_activation, 
                kernel_initializer=init, 
                name=f'encoder_{nst - 1}'
            )

        # decoder

        self.ae_decoder_layers = []

        for i in range(nst - 1, 0, -1):
            x = Dense(
                    layers_half[i], 
                    activation=dec_activation, 
                    kernel_initializer=init, 
                    name=f'decoder_{i}'
                )
            self.ae_decoder_layers.append(x)

        # output
        
        self.ae_output_layer = Dense(
                layers_half[0],
                activation=last_activation,
                kernel_initializer=init,
                name='decoded'
            )

    # ......................... #

    def call(self, inputs, **kwargs):

        x = inputs

        for layer in self.ae_encoder_layers:
            x = layer(x)

        x = self.ae_latent(x)

        for layer in self.ae_decoder_layers:
            x = layer(x)

        y = self.ae_output_layer(x)

        return y

    # ------------------------- #

    def slice_encoder(self) -> Model:

        enc = Sequential()

        for layer in self.ae_encoder_layers:
            enc.add(layer)

        enc.add(self.ae_latent)

        return enc

# ------------------------- #

class DeepAutoRec(Model):

    def __init__(

            self,
            data_shape: Tuple[int],
            layers_dr: Tuple[int] = (256, 512, 256),
            activation_dr: Union[str, Callable] = 'selu',
            last_activation: Union[str, Callable] = 'selu',
            dropout_rate: float = 8e-1,
            l2_enc_rate: float = 1e-3,
            l2_dec_rate: float = 1e-3,
            noise_rate: float = 8e-2,
            **kwargs

        ):

        super().__init__(**kwargs)
        
        # dropout Noise as input layer

        self.dar_noise = Dropout(rate=noise_rate, input_shape=(data_shape[1],))
        
        # encoder

        self.dar_encoder_layers = []

        k = int(len(layers_dr)/2)

        for i, l in enumerate(layers_dr[:k]):

            x = Dense(
                    units=l, 
                    activation=activation_dr,
                    name=f'encoder_{i + 1}', 
                    kernel_regularizer=l2_reg(l2_enc_rate)
                )

            self.dar_encoder_layers.append(x)

        # latent Space

        self.dar_latent = Dense(
                units=layers_dr[k], 
                activation=activation_dr, 
                name='latent_space', 
                kernel_regularizer=l2_reg(l2_enc_rate)
            )
        
        # dropout

        self.dar_dropout = Dropout(rate=dropout_rate)
        
        # decoder

        self.dar_decoder_layers = []

        for i, l in enumerate(layers_dr[k + 1:]):
            x = Dense(
                    units=l, 
                    activation=activation_dr,
                    name=f'decoder_{i + 1}',
                    kernel_regularizer=l2_reg(l2_dec_rate)
                )
            self.dar_decoder_layers.append(x)

        # output layer

        self.dar_output_layer = Dense(
                units=data_shape[1], 
                activation=last_activation, 
                name='predict', 
                kernel_regularizer=l2_reg(l2_dec_rate)
            )

    # ......................... #

    def call(self, inputs, **kwargs):

        x = self.dar_noise(inputs)

        for layer in self.dar_encoder_layers:
            x = layer(x)

        x = self.dar_latent(x)
        x = self.dar_dropout(x)

        for layer in self.dar_decoder_layers:
            x = layer(x)

        y = self.dar_output_layer(x)

        return y

# ------------------------- #

class DeepClustering(Model):

    DEFAULT_CONFIG = {

        'layers_half'  : (100, 80, 50),
        'enc_activation'  : 'relu',
        'dec_activation'  : 'relu',
        'last_activation' : 'selu',
        'learning_rate'   : 1e-5,
        'loss'            : 'mse'

    }

    # ......................... #

    def __init__(

            self,
            data_shape: Tuple[int],
            n_clusters: int = 20,
            config: Dict[str, Any] = dict(),
            with_generator: bool = False,
            **kwargs

        ):

        super().__init__(**kwargs)

        config = dict({**self.DEFAULT_CONFIG, **config})
        config.update({
            'data_shape' : data_shape,
            'latent_space_size' : n_clusters
        })

        lr = config.pop('learning_rate')
        loss = config.pop('loss')

        self.n_clusters = n_clusters
        self.with_generator = with_generator

        self.autoencoder = AutoEncoder(**config)
        self.autoencoder.compile( 
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=loss
            )
        self.cllayer = ClusteringLayer(
                n_clusters=n_clusters,
                name='clustering'
            )
        self.cllayer.build( input_shape=(None, n_clusters) )

    # ......................... #

    def call(self, inputs, **kwargs):

        x = inputs

        for layer in self.autoencoder.ae_encoder_layers:
            x = layer(x)

        x = self.autoencoder.ae_latent(x)
        y = self.cllayer(x)

        return y

    # ......................... #

    def fit(

            self,
            data,
            iter: int = 15000,
            update_interval: int = 300,
            epochs: int = 300,
            batch_size: int = 128,
            batch_size_fine: int = 1024,
            #batch_size: int = 256,
            tolerance: float = 5e-3,
            **kwargs

        ):

        # firstly fit the autoencoder

        if self.with_generator:
            self.autoencoder.fit(
                data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[KerasTqdmBar()], 
                verbose=0
            )
        
        else:
            self.autoencoder.fit(
                data, data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[KerasTqdmBar()], 
                verbose=0
            )
        
        self.autoencoder.trainable = False

        # next fit deepclustering layer

        index = 0
        #loss = 0

        try:
            data = data.to_numpy()

        except:
            pass

        # starts with kmeans initialization and go deeper
        # place using CPU instead of GPU
        kmeans_init = KMeans(
                n_clusters=self.n_clusters, 
                n_init=20, 
                verbose=0
            )
        
        encoder = self.autoencoder.slice_encoder()

        y_pred = kmeans_init.fit_predict(encoder.predict(data))
        y_pred_last = np.copy(y_pred)

        (self
            .get_layer(name='clustering')
            .set_weights([kmeans_init.cluster_centers_])
        )

        index_array = np.arange(data.shape[0])

        # deep training process

        for ite in tqdm(range(iter), bar_format=bar_format):

            if ite % update_interval == 0:

                q = self.predict(data, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                
                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)

                log.info(f'{delta_label=}')

                if ite > 0 and delta_label < tolerance:
                    break

            idx = index_array[index * batch_size_fine: min((index+1) * batch_size_fine, data.shape[0])]
            self.train_on_batch(x=data[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size_fine <= data.shape[0] else 0

    # ......................... #

    @staticmethod
    def target_distribution(x):

        weight = x ** 2 / x.sum(0)
        return (weight.T / weight.sum(1)).T

# ------------------------- #