from tqdm.keras import TqdmCallback

# ------------------------- #

class KerasTqdmBar(TqdmCallback):
    
    def __init__(self, **kwargs):
        kwargs['bar_format'] = '{l_bar}{bar:40} {n_fmt}/{total_fmt} {unit} | ETA: {remaining}'
        super().__init__(**kwargs)

# ------------------------- #