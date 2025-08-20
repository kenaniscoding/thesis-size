from picamera2 import Picamera2

picam2 = Picamera2()

# Configure the camera for 1920x1080 resolution
config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

picam2.start()
picam2.capture_file("img/photo.png")
picam2.stop()
