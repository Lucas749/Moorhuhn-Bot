# Moorhuhn Bot
The bot plays Moorhuhn by searching red pixels and using the k-Means-Algorithm to determine the positions of the birds. Parameters of kMeans, the pixel detection, and the actual playing algorithm were optimized in over 950 games. For score recognition, a neural network and SVM are used that were trained on hand labeled scores and timer values (top left in the game - labeling could be automized this way). However, accuracy is quite low due to insufficient training data.

A video of the bot in action is linked to the picture below.

[![YT Link](https://github.com/Lucas749/Moorhuhn-Autoclicker/blob/master/README%20Pictures/Youtube%20Link.JPG)](https://www.youtube.com/watch?v=HsXdwFAUP_k)

# Files
1. Moorhuhn_Bot.py - Bot that plays Moorhuhn
2. Score_Classifier.py - Generation of score samples and fitting of the neural network and SVM
3. Fitted Models folder - Contains the fitted neural network and SVM
4. Samples Examples folder - Contains a few score samples that were used to train the classifiers

# How it works
1. Create a screenshot of the game
![Screen](https://github.com/Lucas749/Moorhuhn-Autoclicker/blob/master/README%20Pictures/Moorhuhn%20Screen.JPG)

2. Determine red pixels of the birds
![Pixel Detection](https://github.com/Lucas749/Moorhuhn-Autoclicker/blob/master/README%20Pictures/Pixel%20Detection.png)

3. Cluster pixels with KMeans to determine shooting positions
![KMeans Pixel](https://github.com/Lucas749/Moorhuhn-Autoclicker/blob/master/README%20Pictures/Pixel%20KMeans.png)

