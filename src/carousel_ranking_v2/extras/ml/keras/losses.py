import keras.backend as K

# ------------------------- #

def MMSE(mv: float = 0):

    """
    """
    
    def loss(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mv), K.floatx())
        loss = K.square(mask * (y_true - y_pred))
        res = K.sum(loss, axis=-1) / K.sum(mask, axis=-1)

        return res

    loss.__name__ = f'mmse_{mv}'

    return loss

# ------------------------- #