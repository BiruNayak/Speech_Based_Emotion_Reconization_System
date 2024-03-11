
import keras
import librosa
import numpy as np
import pathlib
import os

working_dir_path = pathlib.Path().absolute()

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = os.path.join(working_dir_path , "app","model","Emotion_Voice_Detection_Model.h5") 
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sample_rate = librosa.load(self.file)

        data_trimmed, _ = librosa.effects.trim(data, top_db=20)
    

        # noise cancelation
        X = librosa.effects.preemphasis(data_trimmed)


        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

        # Compute delta and delta-delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
        # Store features in a list
        feature_list = [mfccs, delta_mfccs, delta_delta_mfccs]
        x = np.expand_dims(feature_list, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict(x)
        
        return self.convert_class_to_emotion(predictions)

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}
        
        for key, value in label_conversion.items():
           if int(key) == np.argmax(pred):
               label = value
        return label