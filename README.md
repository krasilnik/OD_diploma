Automated System for Determining the Coordinates and Type of Moving Object from a Quadcopter
To use the system, you need to:

Download the necessary files (best.pt, main2).
Download all the required libraries mentioned in main2.
For testing the system, you can use any video with russian soldiers or military vehicles filmed from a drone. It is recommended to keep all files in one folder.
To test the distance determination algorithm, you can use the files main and faces.pt, repeating the steps for using the system as mentioned above. The distance determination algorithm is tested in real-time using a camera, and the algorithm determines the distance to your face. It is recommended to change the Focal_length variable in the main file to the focal length of your camera for optimal performance.

The file rus_detect contains code for training, validation, and testing the model.

To view all the images used in the model or if you want to download the dataset, follow the link below.

https://universe.roboflow.com/krasilnik-ivan/rus-detect






