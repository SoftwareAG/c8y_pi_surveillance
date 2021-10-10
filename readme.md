# Movement detection and persons counter for Raspberry PI

This Python program works with Python 3.
It includes pretrained model.
A Configuration file called "conf.json" needs to be added in the app folder.
Content should look like that:
```
{
	"show_video": true,
	"min_upload_seconds": 3.0,
	"min_motion_frames": 8,
	"camera_warmup_time": 2.5,
	"delta_thresh": 5,
	"resolution": [640, 480],
	"fps": 16,
	"min_area": 5000,
	"c8y_baseurl": "https://<yourtenantname>.cumulocity.com",
	"c8y_username": "<yourusername>",
	"c8y_password": "<yourpassword>"
}
```
You will need to install the following packages (sudo apt-get install):
- python3
- libopencv-dev
- python3-opencv
- pyhton3-scipy
- libhdf5-dev
- libhdf5-serial-dev
- libcblas-dev
- libatlas-base-dev
- libjasper-dev
- libqtgui4
- libqt4-test

Then install these Python libraries (pip3 install):
- imutils
- numpy
- pybind11
