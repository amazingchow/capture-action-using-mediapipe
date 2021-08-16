# capture-action-using-mediapipe

capture dance using human gesture recognition tool -- mediapipe powered by google 

## Get Started

### Prerequisites

```text
* Python3.8+
```

### Installation

#### Clone

* Clone this repo to your local machine using https://github.com/amazingchow/capture-action-using-mediapipe.git.

#### Setup

```shell
# (optional) If use the capture-action-using-mediapipe on Ubuntu, install freetype lib first.
➜ sudo apt-get install libfreetype6-dev
# (optional) If not found libfreetype6-dev, just find a alternative one.
➜ apt-cache search freetype | grep dev

# Setup a python virtual environment
➜ cd /path/to/capture-action-using-mediapipe
➜ virtualenv -p /usr/bin/python3 .opencv-pyvenv
➜ source .opencv-pyvenv/bin/activate

# Install third-party dependencies
(opencv-pyvenv) ➜ pip install numpy
(opencv-pyvenv) ➜ pip install matplotlib
(opencv-pyvenv) ➜ pip install opencv-contrib-python
(opencv-pyvenv) ➜ pip install mediapipe

# Run the script
(opencv-pyvenv) ➜ python pose_recognition_from_camera.py --video_file=/path/to/your_video.mp4
```

## Reference

* [mediapipe-python-solution-api](https://google.github.io/mediapipe/solutions/pose#python-solution-api)

## Contributing

### Step 1

* 🍴 Fork this repo!

### Step 2

* 🔨 HACK AWAY!

### Step 3

* 🔃 Create a new PR using https://github.com/amazingchow/capture-action-using-mediapipe/compare!

## Support

* Reach out to me at <jianzhou42@163.com>.

## License

* This project is licensed under the MIT License - see the **[MIT license](http://opensource.org/licenses/mit-license.php)** for details.