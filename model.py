# pip install -r requirements.txt

import pandas as pd
import joblib

class Model:

    def __init__(self):
        self.model = joblib.load('static/model.joblib')
        self.scaler = joblib.load('static/scaler.joblib')
        self.feature_names = ["cap-diameter", "cap-shape", "gill-attachment", "gill-color", 
                              "stem-height", "stem-width", "stem-color", "season"]

    def predict(self, cap_diameter, cap_shape, gill_attachment, gill_color, stem_height, stem_width, stem_color, season):

        data = pd.DataFrame([[cap_diameter, cap_shape, gill_attachment, gill_color, 
                              stem_height, stem_width, stem_color, season]], 
                            columns=self.feature_names)
        
        scaled_data = self.scaler.transform(data)
        y_pred = self.model.predict(scaled_data)[0]
        return int(y_pred)



