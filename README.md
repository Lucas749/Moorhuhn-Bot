# Moorhuhn Bot
The bot plays Moorhuhn by searching red pixels and using the k-Means-Algorithm to determine the positions of the birds. Parameters of kMeans, the pixel detection, and the actual playing algorithm were optimized in over 950 games. 
For score recognition, a neural network is used that was trained on hand labeled scores and timer values (top left in the game - labelling could be automized this way). However, the accuracy is quite low due to insufficient training data.

Project includes the following files:
1. Moorhuhn_Bot.py - Bot that plays Moorhuhn
2. Score_Classifier.py - Generation of score samples and fitting of the neural network

