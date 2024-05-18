import cv2
import numpy as np
import time

class Video(object):
    def __init__(self):
        self.dirname = ""
        self.cap = None
        self.t0 = 0
        self.frame_count = 0
        self.valid = False
        self.shape = None

    def start(self):
        print("Start video")
        if not isinstance(self.dirname, str) or not self.dirname:
            print("Invalid filename!")
            self.valid = False
            return

        self.cap = cv2.VideoCapture(self.dirname)
        if not self.cap.isOpened():
            print(f"Cannot open video source: {self.dirname}")
            self.valid = False
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.t0 = time.time()
        
        try:
            resp, frame = self.cap.read()
            if not resp:
                print("Failed to read the video")
                self.valid = False
            else:
                self.shape = frame.shape
                self.valid = True
        except Exception as e:
            print(f"Error reading frame: {e}")
            self.valid = False

        if self.valid:
            print(f"FPS: {fps}")
            print(f"Frame count: {self.frame_count}")
            print(f"Video start time: {self.t0}")
            print(f"Frame shape: {self.shape}")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("Stop video")
            self.cap = None
            self.valid = False

    def get_frame(self):
        if self.valid:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("End of video")
                self.stop()
                print(f"Total time: {time.time() - self.t0} seconds")
                return None
            else:
                try:
                    frame = cv2.resize(frame, (640, 480))
                except Exception as e:
                    print(f"Error resizing frame: {e}")
                    self.stop()
                    return None
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Cannot load the video)", (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

# Example usage
if __name__ == '__main__':
    video = Video()
    video.dirname = r"C:\Users\hp\Downloads\Heart-rate-measurement-using-camera-master\vid.mp4"  # Set the video file path here
    video.start()
    while video.valid:
        frame = video.get_frame()
        if frame is None:
            break
        # Display the frame or process it as needed
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.stop()
    cv2.destroyAllWindows()
