from unicodedata import category
import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLE_TO_CONSIDER=22050  # 1 sec worth of sound

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # data dictionary
    data = {
        "mappings":[],
        "labels":[],
        "MFCCs":[],
        "files":[]
    }

    # loop through all the sub-dirs
    for i,(dirpath,dirnames,filenames) in enumerate(os.walk(DATASET_PATH)):

        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # extract MFCCs
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath,f)

                # load audio file
                signal,sr = librosa.load(file_path)

                # ensure the audio file is at lease 1 sec
                if len(signal)>= SAMPLE_TO_CONSIDER:
                    
                    # enforce 1 sec. long signal
                    signal = signal[:SAMPLE_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal,n_mfcc=n_mfcc,hop_length=hop_length,n_fft=n_fft)
                    
                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    # store in json file
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

if __name__=="__main__":
    prepare_dataset(DATASET_PATH,JSON_PATH,n_mfcc=13, hop_length=512, n_fft=2048)

