import os
import librosa
import math
import json

DATASET_PATH = 'genres_original'
JSON_PATH = 'data.json' # output path
SAMPLE_RATE = 22050
DURATION = 30 #seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 10): # Here num_segments means that our audio will be divided into 10 parts

    # dictionary to store data
    data = {"mapping": [], # here we will have the genre names like classical, blues etc.
            "mfcc": [], # here we will store the mfcc coefficients
            "labels": [] # here we will store the correspoding label(integer value) for the genre names
            }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # math.ceil() 1.2 -> 2

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we are not at the root level(top level)
        if dirpath is not dataset_path: # as dirpath first return the main folder name i.e. genres_original_copy(here)
            # save the semantic label
            dirpath_component = dirpath.split('\\') # genres_original_copy/classical => ['genres_original_copy', 'classical']
            semantic_label = dirpath_component[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load the audio file
                file_path = os.path.join(dirpath, f)
                # signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    # 音频处理代码
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    end_sample = start_sample + num_samples_per_segment

                    # mfcc = librosa.feature.mfcc(signal[start_sample:end_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) # This way we used segments
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                    mfcc = mfcc.T

                    # Now some mfcc will have different size as due to the duration the samples are not equally divided so some mfcc may have more or less size than others
                    # So we will store mfcc for segment if it has expected length as all the input to the model should have the same size
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist()) # mfcc returns np array but to save in json file we need to convert it to list
                        data['labels'].append(i-1) # i-1 because the first i is for the given dataset path which we don't have to count
                        print("{}, segment:{}".format(file_path, s))

    # Save the data
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
