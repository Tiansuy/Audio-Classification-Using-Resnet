# Audio-Classification-Using-Resnet

<hr>
<b> Dataset Link :- </b> https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br>
<b> About the data :- </b> Genre original folder(only this required) - It is a collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds).

## Implementation Guide
<ul>
  <li> First of all, we have to do some data preprocessing and extract some useful information from our music data so that we can use it for training our model. For this run:-
   
   ```python prepare_dataset.py```
  </li>
  <li> Next there are 2 custom model(one is CNN based) built using PyTorch and it is trained on the preprocessed data. 
     For CNN based model run:-
  
  ```python audio_cnn_pytorch.py```
     
  </li>

  - with acc 44.5%
</ul>
