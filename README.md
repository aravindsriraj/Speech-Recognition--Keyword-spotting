# Speech-Recognition--Keyword-spotting

A Simple Speech recognition system using Deep learning for keyword spotting like "up","down","left","right" etc

Dataset can be downloaded from https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

The dataset consists of 30 keywords but I've used only 10 keywords for training purpose

## Steps to run the project
1. Download the dataset and store in the same project directory
2. Each sub-directories inside the dataset are the keywords.
3. The keywords I used are ....down, go, left, no, off, on, right, stop, up, yes
4. You can delete the folders(Keywords) that you are not training
5. Run prepare_dataset.py to extract MFCC features and labels from audio data files and convert it to JSON format.
6. Run train.py to train the neural network.
Run keyword_spotting_service.py to run the speech recognition
