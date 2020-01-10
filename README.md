# Moorhuhn Bot
The bot plays Moorhuhn by searching red pixels and using the k-Means-Algorithm to determine the positions of the birds. Parameters of kMeans, the pixel detection, and the actual playing algorithm were optimized in over 950 games. For score recognition, a neural network is used that was trained on hand labeled scores and timer values (top left in the game - labeling could be automized this way). However, accuracy is quite low due to insufficient training data.

# Files
1. Moorhuhn_Bot.py - Bot that plays Moorhuhn
2. Score_Classifier.py - Generation of score samples and fitting of the neural network

# How it works
1. Create a screenshot of the game
![Test Image 1](https://github.com/Lucas749/Moorhuhn-Autoclicker/blob/master/Test.JPG)

2. Determine red pixels of the birds

3. Cluster pixels with KMeans to determine shooting positions

