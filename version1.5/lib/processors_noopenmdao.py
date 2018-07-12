import numpy as np
import time
import cv2
import pylab
import os


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):
        self.num_of_frames = 0
        self.frame_in = np.zeros((10, 10))
        self.all_frames = []
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        self.fps2 = 0
        self.times = []
        self.ttimes = []
        self.fft = []
        self.xy=[]
        self.indexs = []
        self.selected_frames = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        self.t0 = time.time()
        self.num_of_not_find_face = 0
        self.flag = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False
        self.start = 0

        self.idx = 1
        self.idx2 = 1
        self.find_faces = True
        self.tracker = cv2.TrackerMIL_create()

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces



    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    # def draw_rect(self, rect, col=(0, 255, 0)):
    #     x, y, w, h = rect
    #     cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h,bbox):
        x, y, w, h = bbox
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained

    def find_face(self, frame):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                           scaleFactor=1.3,
                                                           minNeighbors=4,
                                                           minSize=(
                                                               50, 50),
                                                           flags=cv2.CASCADE_SCALE_IMAGE))

        if self.num_of_not_find_face > 5:
            cv2.putText(self.frame_out, "Does not recognize the face well enough",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(self.frame_out, "Press 'S' to start again",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            self.num_of_frames = 400
            self.flag = 1
        if self.num_of_frames != 400:
            if len(detected) == 0:
                self.num_of_not_find_face += 1
            if len(detected) > 0:
                self.num_of_not_find_face = 0
                detected.sort(key=lambda a: a[-1] * a[-2])
                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
                if self.start == 0:
                    x1, y1, w1, h1 = self.face_rect
                    bbox = x1, y1, w1, h1
                    ok = self.tracker.init(frame, bbox)
                    self.start = 1
                if set(self.face_rect) == set([1, 1, 2, 2]):
                    return
                ok, bbox = self.tracker.update(frame)
                x, y, w, h = self.get_subface_coord(0.5, 0.16, 0.25, 0.15, bbox)
                forehead1 = x, y, w, h
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 2)
                fobject = (frame, forehead1)
                self.all_frames.append(fobject)
                if len(self.all_frames) > 300:
                    # self.get_selected_frames_x()
                    self.check_big_move()
                    if self.num_of_frames != 400:
                        self.get_selected_frames_y()
                        # self.selected_frames = self.all_frames.copy()
                        # self.run()

    def check_big_move(self):
        i = 0
        temp = []
        while i < len(self.all_frames)-1:
            diff = np.array(self.all_frames[i+1][1][0]) - np.array(self.all_frames[i][1][0])
            temp.append(abs(diff))
            i += 1
        if max(temp) > 20:
            cv2.putText(self.frame_out, "Please don't move so much ",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(self.frame_out, "Press 'S' to start again",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            self.num_of_frames = 400

    # Selects all frames with the closest X
    def get_selected_frames_x(self):
        i = 0
        j = 0
        temp = []
        while i < len(self.all_frames):
            self.xy.append(self.all_frames[i][1][0])
            diff = np.array(self.all_frames[0][1][0]) - np.array(self.all_frames[i][1][0])
            temp.append(abs(diff))
            i += 1
        while j < 200:
            min_x = temp.index(min(temp))
            self.indexs.append(min_x)
            temp[min_x] = np.math.inf
            self.selected_frames.append(self.all_frames[min_x])
            j += 1
        self.run()

    # Selects all frames with the closest X
    def get_selected_frames_y(self):
        i = 0
        j = 0
        temp = []
        while i < len(self.all_frames):
            self.xy.append(self.all_frames[i][1][1])
            diff = np.array(self.all_frames[0][1][1]) - np.array(self.all_frames[i][1][1])
            temp.append(abs(diff))
            i += 1
        while j < 200:
            min_y = temp.index(min(temp))
            self.indexs.append(min_y)
            temp[min_y] = np.math.inf
            self.selected_frames.append(self.all_frames[min_y])
            j += 1
        self.run()

    def run(self):
            self.times.append(time.time() - self.t0)
            xg1 = np.zeros(len(self.selected_frames), 'double')
            xg2 = np.zeros(len(self.selected_frames), 'double')
            xg3 = np.zeros(len(self.selected_frames), 'double')
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for k in range(0, len(self.selected_frames) - 1):
                green_image = self.selected_frames[k][0]
                fox, foy, fow, foh = self.selected_frames[k][1]
                nop = fow * foh
                for x in range(fox, fox + fow-1):
                    for y in range(foy, foy + foh-1):
                        if x < 480 and y < 480:
                            sum1 = (green_image[x, y, 1]) + sum1
                            sum2 = (green_image[x - 15, y, 1]) + sum2
                            sum3 = (green_image[x + 15, y, 1]) + sum3
                xg1[k] = sum1 / nop
                xg2[k] = sum2 / nop
                xg3[k] = sum3 / nop
                sum1 = 0
                sum2 = 0
                sum3 = 0
            xgm1 = xg1 - np.mean(xg1)
            xgm2 = xg2 - np.mean(xg2)
            xgm3 = xg3 - np.mean(xg3)
            Fs = 27
            T = 1 / Fs
            l = len(self.selected_frames)
            myNFFT = 2 ** (np.math.ceil(np.math.log(l, 2)))
            yg1 = np.fft.rfft(xgm1, myNFFT) / l
            yg2 = np.fft.rfft(xgm2, myNFFT) / l
            yg3 = np.fft.rfft(xgm3, myNFFT) / l
            f = Fs / 2 * np.linspace(0, 1, (myNFFT / 2) + 1)
            yg4 = yg1
            y1 = 2 * np.abs(yg1[1:np.math.floor((myNFFT / 2) + 1)])
            y2 = 2 * np.abs(yg2[1:np.math.floor((myNFFT / 2) + 1)])
            y3 = 2 * np.abs(yg3[1:np.math.floor((myNFFT / 2) + 1)])
            index = 0
            f1 = 0
            for a in range(np.math.floor(len(f) / 15 * 0.9 + 1), np.math.floor(len(f) / 15 * 3.5)):
                if f1 < y1[a]:
                    f1 = y1[a]
                    index = a
            f1_old = f1
            index_old = index
            detected_pulse_1 = 60 * (index - 1) * 15 / (len(y1) - 1)
            pulse_amplitude_1 = y1[index]
            print("det")
            print(detected_pulse_1)
            col = (255,0, 0)
            if detected_pulse_1 >= 60 and detected_pulse_1 <= 121:
                cv2.putText(self.frame_out, "pulse exist",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,  col, 2, cv2.LINE_AA)
                print("pulse exist")
                self.num_of_frames = 300
            else:
                cv2.putText(self.frame_out, "no pulse",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,  col, 2, cv2.LINE_AA)
                print("no pulse!")
                self.num_of_frames = 300

