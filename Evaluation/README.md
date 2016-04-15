#ActivityNet Large Scale Activity Recognition Challenge - Evaluation Toolkit
This is the documentation of the ActivityNet Large Scale Activity Recognition
Challenge Evaluation Toolkit. It includes APIs to evaluate the performance of a method in the two different tasks in the challenge: *untrimmed video classification* and *activity detection*. For more information about the challenge competitions, please read the [guidelines](http://activity-net.org/challenges/2016/guidelines.html).

##Dependencies
The Evaluation Toolkit is purely written in Python (>=2.7) and it requires the 
following third party libraries:
* [Numpy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)

##Getting started
We include sample prediction files in the folder data to show how to evaluate your prediction results. Please follow this steps to obtain the performance evaluation on the provided sample files:
* Run `git clone` this repository.
* To evaluate classification performance call: `python get_classification_performance.py data/activity_net.v1-3.min.json sample_classification_prediction.json`
* To evaluate detection performance call: `python get_detection_performance.py data/activity_net.v1-3.min.json sample_detection_prediction.json`

##Contributions and Troubleshooting
We are welcome to contributions, please keep your pull-request simple so we can go back to you as soon as we can. If you found a bug please open a new issue and describe the problem.
