import os
import shutil
import glob
import librosa
import shutil
import random
import soundfile as sf

from config import RAVDESS_ORIGINAL_FOLDER_PATH, TESS_ORIGINAL_FOLDER_PATH, INTEGRATED_AUDIO_PATH

"""
This file builds 2 additional actor folders (25 and 26) using features from the
Toronto emotional speech set (TESS) dataset: https://tspace.library.utoronto.ca/handle/1807/24487

These stimuli were modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966).
A set of 200 target words were spoken in the carrier phrase "Say the word _____'
by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions
(anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.
Two actresses were recruited from the Toronto area. Both actresses speak English as their first language,
are university educated, and have musical training. Audiometric testing indicated that
both actresses have thresholds within the normal range.

Authors: Kate Dupuis, M. Kathleen Pichora-Fuller

University of Toronto, Psychology Department, 2010.

TESS data can be downloaded from here: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess/data

To facilitate the feature creation, the TESS data have been renamed using the same naming convention adopted
by the RAVDESS dataset explained below:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

In case of TESS files, an example below. We do not care of assigning values other than the ones
specified below as those are not used by the model, hence we are assigning random integers.
- 03 (Random)
- 01 (Random)
- 01 (This varies according to the fact in TESS we have 1 emotion less then RAVDESS: calm).
- 01 (Random)
- 03 (Random).
- 01 (Random)
- 01 (Random. I thought initially to put 25 if YAF, 26 if OAF, but that is not needed as the pipeline is not
using the actor information from the filename, only the mfccs extracted from librosa and the target emotion).
"""




class TESSPipeline:

    @staticmethod
    def create_tess_folders(path):
        """
        Create Actor_25 and Actor_26 folders and copy/rename TESS audio files accordingly.
        """
        # Label conversion for emotions
        label_conversion = {'01': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'ps'}

        # Iterate through TESS audio files
        for subdir, _, files in os.walk(path):
            for filename in files:
                if filename.startswith('OAF'):
                    actor_folder = 'Actor_26'
                else:
                    actor_folder = 'Actor_25'

                destination_path = os.path.join(INTEGRATED_AUDIO_PATH, actor_folder)
                os.makedirs(destination_path, exist_ok=True)

                old_file_path = os.path.join(subdir, filename)

                # Separate base from extension
                base, extension = os.path.splitext(filename)

                for key, value in label_conversion.items():
                    if base.endswith(value):
                        random_list = random.sample(range(10, 99), 7)
                        file_name = '-'.join([str(i) for i in random_list])
                        
                        # Construct file name with emotion code
                        if actor_folder == 'Actor_26':
                            file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                        else:
                            file_name_with_correct_emotion = (file_name[:6] + key + file_name[8:] + extension).strip()

                        new_file_path = os.path.join(destination_path, file_name_with_correct_emotion)
                        shutil.copy(old_file_path, new_file_path)


class DataIntegration:
    @staticmethod
    def load_ravdess_data(folder_path):
        """
        Load RAVDESS audio files from the specified folder path.

        Args:
            folder_path (str): The path to the RAVDESS dataset folder.

        Returns:
            list: List of file paths for RAVDESS audio files.
        """
        ravdess_data = []
        # Loading RAVDESS audio files
        audio_files = glob.glob(os.path.join(folder_path, '*', '*.wav'))
        for audio_file in audio_files:
            if not audio_file.endswith('.DS_Store'):
                ravdess_data.append(audio_file)
        return ravdess_data

    @staticmethod
    def preprocess_ravdess_data(audio_files, output_directory):
        """
        Preprocess RAVDESS audio files by trimming silence and saving to the output directory.

        Args:
            audio_files (list): List of file paths for RAVDESS audio files.
            output_directory (str): The path to the directory where trimmed audio will be saved.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Iterate through each audio file
        for audio_file in audio_files:
            # Load the audio file
            y, sr = librosa.load(audio_file)
            
            # Trim the audio
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # Extract actor number from the filename (assuming the format is '03-01-02-01-02-01-16.wav')
            actor_number = int(audio_file.split('-')[-1].split('.')[0])

            # Create the output directory for the specific actor if it doesn't exist
            actor_output_directory = os.path.join(output_directory, f'Actor_{actor_number}')
            os.makedirs(actor_output_directory, exist_ok=True)

            # Generate the output filename
            output_filename = os.path.join(actor_output_directory, os.path.basename(audio_file))

            # Save the trimmed audio to the output directory using soundfile
            sf.write(output_filename, y_trimmed, sr)


if __name__ == '__main__':
    # Preprocess RAVDESS data
    ravdess_data = DataIntegration.load_ravdess_data(RAVDESS_ORIGINAL_FOLDER_PATH)
    DataIntegration.preprocess_ravdess_data(ravdess_data, INTEGRATED_AUDIO_PATH)

    # Convert and save TESS data using TESSPipeline
    TESSPipeline.create_tess_folders(TESS_ORIGINAL_FOLDER_PATH)
