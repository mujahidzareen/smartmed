from sklearn.preprocessing import LabelEncoder
import numpy as np

class ExtendedLabelEncoder(LabelEncoder):
    print("I am inside the encoder function")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, y):
        unseen_labels = set(y) - set(self.classes_)
        if unseen_labels:
            self.classes_ = np.append(self.classes_, list(unseen_labels))
        return super().transform(y)
