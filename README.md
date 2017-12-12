# pig_identification
This is the subject of JoyBuy AI Competition 2017.

## Introduction
The network build with deep residual netowrk from Facebook, and inspired by Deep Networks with Stochastic Depth, adding drop out inside the residual block, to improve the accuracy.

## The Dataset
The data set contains 30 videos (1280x720 30fps) of pig, each video indicates a pig, and duration of each average at 90 seconds. The test set contains 3000 jpg images.

##Preprocess
The video had been convert into 120x64, and use center crop function to cut off the video into 64X64. the label use for one hot encode. 70% of video frame use for training, and 10% use for validating, and rest of them use for testing.

