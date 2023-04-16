# GestureWave

## Hardware details 
### samsung galaxy watch 4 

# Installing dependencys 
### 1) Go to  `GestureWave` directory 
### 2) excute `pip3 install -r requirements.txt` in the terminal 



## Steps to install the watch application 
### 1) Open the `watch_application` directory in android studio. 
### 2) Enable wifi debugging in the watch's developer setting.
### 3) [Install the adb tool](https://www.xda-developers.com/install-adb-windows-macos-linux/)
### 4) make sure the watch is linked to the same network as the device. 
### 5) The network's ip address is presented under the option "wifi debugging" in the watch developer menu. 
### 6) In the terminal of the android studio excute `adb connect <ip address of the network>`
### 7) Now that the watch and Android Studio are linked, simply click `build` and `run`.

## Steps to run the `server.py` from another device connected to the same network
### 1) Go into the `server` directory 
### 2) excute `python server.py`

### When both the server.py and the watch_application are active, a socket connect connection is made between them.


# How to train machine learning modal 
..

#

# setup the website

### 1) cd watch-ui
### 2) npm install, install all the required dependencies from package.json
### 3) npm run dev, start the server on port 3000
