# Object-Tracking
It tracks the object in a moving background using Python and Open Cv. 
The xml contains the detected coordinates of the object. Object is successfully tracked using a algorithm designed. 
The algorithm uses LBP and PHOG feature vectors for tracking. 
The feature vectors help in calculating the similarity score on basis of which most probable position of object is decided.
The score is calculated after the euclidean distance from refrence vector is calculated and put in the sigmoid function.
