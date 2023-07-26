from DanceAnalyzer import danceanalyzer
import sys

# example:
# python dance-app.py samples\resources\danceexample.mp4

if len(sys.argv) < 2:
    print("USAGE: dance-app.py [video_location]")
else:
    app = danceanalyzer()
    file_location, csv_location = app.analyzeVideo(sys.argv[1])
    app.liveWebCam(file_location, csv_location)