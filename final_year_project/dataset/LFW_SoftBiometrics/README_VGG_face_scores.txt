VGG-face scores associated to the 10 folds from view 2 of LFW database

Face features are a 4096-feature vector obtained by feedforwarding the LFW images across
The VGG-face pre-trained model.

There are 20 different files, 2 per each fold (genuine and impostor).

File Format

vggface_numfold_[genuine/impostor], where numfold ranges from 01 to 10

In each file, we will find

File1 File2 OutputScore1 OutputScore2

Where:
	-OutputScore1: score using features extracted from the fully connected layer 6 (fc6) and cosine distance
	-OutputScore2: score using features extracted from the fully connected layer 7 (fc7) and cosine distance