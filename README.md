# PylonCameraControl

PylonCameraControl is a GUI-based software to control the Basler Camera to capture hologram of water sample and control the Arudino which connect to the pump and the LED flash light.

 - Use GPU/CPU to decode and process the byte stream from camera to different types of raw images 
 - With SSD hard drive and GPU mode, can achieve processing and saving over 60MB image file per 0.3 sec with no significant memory growth (if SSD is big enough, it can run forever)
 - Sync the camera capturing and LED flash light

### To build the project you need:
 - Install Pylon SDK from [Software Downloads | Basler](http://www.baslerweb.com/en/support/downloads/software-downloads?type=27&series=0&model=0)
 - Install QT 5.x and Qt Visual Studio Add-in
 - Install CUDA 7.5 (or manully change the CUDA setup in .vcxproj file to your version)
 - Download the resource file from [idx_map.csv](https://drive.google.com/open?id=0B5iFpirth8VfQ1dUdThwc1JTNFE) and put into project folder

### File Description
 - <code>main</code>: Entry point.  
 - <code>PylonCameraWindow.ui</code>: GUI designed with Qt.  
 - <code>PylonCameraWindow</code>: Logic part for the GUI.  
 - <code>PylonControl</code>: Main thread running back-end to do all the stuff.  
 - <code>PylonCameraHelper</code>: Helper class to initialize and control the camera. Pass the byte stream to the background workers for further process.  
 - <code>WriteFileWorker</code>: Background thread to process the image with CPU (usually cost 120ms) and write files.   
 - <code>CUDAWriteFileHelper</code>: Helper class for launching CUDA kernels. Process the image with GPU (usually cost 22ms, thread blocking) and write files (not blocking).  
 - <code>CudaImgPreprocessKernels</code>: CUDA kernel functions to decode the byte stream and transform to the desired format.  
 - <code>ArduinoControlHelper</code>: Initiallize and write command to Arduino using QSerialPort.  
 - <code>Timer</code>: A chrono timer to get timestamp.  
 


