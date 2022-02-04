# Audio enhancement using deep neural network

This project done for the deep learning course of master 2 SIGMA at Grenoble INP university.

The aim of this project was to design a simple neural net to be abale to remove a low frequency noise from audio. 

The decision made to use classic autoencoders and CNN autoencoders on STFT of the audio signal by considering the STFT as an image and use image denoising techniques. The dataset was created by splitting each STFT to 16 frames times the size of the windows.

Also, in the beginning, we implemented a classical model called IBM separation with this [github repository](https://github.com/simonsuthers/IBM-Separation.git).

All the materials and datasets can be downloaded from my [google drive directory](https://drive.google.com/drive/folders/1wmcxBIE5IkBUVcQVXWjDWGP7Bd4O2CeF).
