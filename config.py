import pathlib
import os

working_dir_path = pathlib.Path().absolute()


RAVDESS_ORIGINAL_FOLDER_PATH = os.path.join(str(working_dir_path), 'Datasets','Original_data','RAVDESS','Audio_Speech_Song_Actors_01-24')
TESS_ORIGINAL_FOLDER_PATH = os.path.join(str(working_dir_path), 'Datasets','Original_data','TESS Toronto emotional speech set data')
INTEGRATED_AUDIO_PATH = os.path.join(str(working_dir_path),'Datasets','Processed Data')
SAVE_DIR_PATH = os.path.join(str(working_dir_path), 'joblib_features')
MODEL_DIR_PATH = os.path.join(str(working_dir_path), 'model')
EXAMPLE_PATH = os.path.join(str(working_dir_path), 'examples')
RESULT_DIR_PATH = os.path.join(str(working_dir_path), 'results')