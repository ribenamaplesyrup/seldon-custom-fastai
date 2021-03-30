from fastai.tabular.all import load_learner
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Union, Iterable
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

logger = logging.getLogger(__name__)

class AdultClassifier(object):

    def __init__(self):
        self.ready = False

    def load(self):
        logger.info('Loading...')
        self.model = load_learner('model/tabular.pkl')
        self.ready = True

    def predict(self, X: List,
                features_names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        try:
            if not self.ready:
                self.load()

            logger.info("Calling predict...")
            series = pd.Series(X, features_names)
            prediction = self.model.predict(series)
            return prediction[2:3][0].tolist()

        except Exception as ex:
            logging.exception("Exception during predict!")
            logging.exception(f"{ex}")
