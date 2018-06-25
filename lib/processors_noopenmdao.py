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
        # self.window = np.hamming(self.buffer_size)
        self.fps2 = 0
        self.data_buffer = []
        self.data_buffer2 = []
        self.data_buffer3 = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        # self.times2 = []
        # self.ttimes2 = []
        self.samples2 = []
        self.freqs2 = []
        self.fft2 = []
        self.slices2 = [[0]]
        self.t0 = time.time()
        self.bpms2 = []
        self.bpm2 = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.idx2 = 1
        self.find_faces = True
        self.sum1=0

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

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
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

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):
        print(self.num_of_frames)
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                        (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            self.data_buffer, self.times, self.trained = [], [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])

                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
            forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)
        # vals = self.get_subface_means(forehead1)

        # self.data_buffer.append(vals)

        def nextpow2(i):
            n = 1
            while n < i:
                n *= 2
            return n
        xg1 = np.zeros(self.num_of_frames - 19, 'double')
        xg2 = np.zeros(self.num_of_frames - 19, 'double')
        xg3 = np.zeros(self.num_of_frames - 19, 'double')
        self.sum1 = 0
        sum2 = 0
        sum3 = 0
        fox, foy, fow, foh = forehead1

        green_image = self.frame_in.copy()  # Make a copy
        green_image[:,:,0] =0
        green_image[:,:,2]=0
        nop = fow * foh
        if self.num_of_frames > 19:
            for k in range(1, self.num_of_frames-19):
                green_image=self.all_frames[k]
                for x in range(fox, fox+fow):
                    for y in range(foy, foy+foh):
                        self.sum1 = (green_image[x, y,1]) + self.sum1
                        sum2 = (green_image[x-15, y,1]) + sum2
                        sum3 = (green_image[x+15, y,1]) + sum3
                xg1[k] = self.sum1 / nop
                xg2[k] = sum2 / nop
                xg3[k] = sum3 / nop
                self.sum1 = 0
                sum2 = 0
                sum3 = 0
            print("xg1111")
            print(xg1)
            xgm1 = xg1 - np.mean(xg1)
            xgm2 = xg2 - np.mean(xg2)
            xgm3 = xg3 - np.mean(xg3)
            print("xgm")
            print(xgm1)
            full_time_array_1 = xgm1
            full_time_array_2 = xgm2
            full_time_array_3 = xgm3
            Fs = 27
            T = 1 / Fs
            l = self.num_of_frames - 19
            myNFFT = 2 ^ nextpow2(l)
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
            if detected_pulse_1>= 60 and detected_pulse_1 <= 120:
                print("pulse exist")
                self.num_of_frames = 2000
            else:
                print("no pulse!")
            # finding the next pulse in the main roi
            index2 = 0
            pick=y1[index]
            f1 = 0
            y1[index]=0
            for b in range(np.math.floor(len(f) / 15 * 0.9 + 1), np.math.floor(len(f) / 15 * 3.5)):
                if f1 < y1[b]:
                    f1 = y1[b]
                    index2 = b
            nextPulse_1 = 60 * (index2 - 1) * 15 / (len(y1) - 1)
            next_pulse_amplitude_1 = y1[index2]
            print("nextPulse")
            print(nextPulse_1)


        #this is the improvement of the algorithm that compare 3 overlapping areas,
        #and finds peak that exists in all of them

        for a in range(np.math.floor(len(f) / 15 * 0.9 + 1), np.math.floor(len(f) / 15 * 3.5)):
            if (np.abs(yg1[a]) > np.abs(yg1[a-2]) and (yg1[a]) > np.abs(yg1[a+2]))and (np.abs(yg2[a]) > np.abs(yg2[a-2])and(yg2[a])>np.abs(yg2[a+2])) and(np.abs(yg3[a])>np.abs(yg3[a-2])and (yg3[a])>np.abs(yg3[a+2])):
                y = max(np.abs(yg3[a]), np.abs(yg1[a]))
                yg4[a] = max(y, np.abs(yg2[a]))
            else:
                x = min(np.abs(yg3[a]), np.abs(yg1[a]))
                yg4[a] = min(np.abs(yg2[a]), x)
        y4 = 2*np.abs(yg4[1:np.math.floor((myNFFT / 2) + 1)])
        f11 = 0
        for a in range(np.math.floor(len(f) / 15 * 0.9 + 1), np.math.floor(len(f) / 15 * 3.5)):
            if f11 < y4[a]:
                f11 = y4[a]
                index1 = a

        f11_old = f11
        index1_old = index1
        detected_pulse_2 = 60 * (index1 - 1) * 15 / (len(y4) - 1)
        pulse_amplitude_2 = y4[index1]
        print("det-after")
        print(detected_pulse_2)
        print("amplitdue")
        print(pulse_amplitude_2)

        # xf, yf, wf, hf = forehead1

        # forehead2 = [xf+15, yf, wf, hf]
        # self.draw_rect(forehead2)
        # forehead3 = [xf - 15, yf, wf, hf]
        # self.draw_rect(forehead3)

        # L = len(self.data_buffer)
        # if L > self.buffer_size:
        #     self.data_buffer = self.data_buffer[-self.buffer_size:]
        #     self.times = self.times[-self.buffer_size:]
        #     L = self.buffer_size
        #
        # processed = np.array(self.data_buffer)
        # self.samples = processed
        # if L > 10:
        #     self.output_dim = processed.shape[0]
        #
        #     self.fps = float(L) / (self.times[-1] - self.times[0])
        #     even_times = np.linspace(self.times[0], self.times[-1], L)
        #     interpolated = np.interp(even_times, self.times, processed)
        #     interpolated = np.hamming(L) * interpolated
        #     interpolated = interpolated - np.mean(interpolated)
        #     raw = np.fft.rfft(interpolated)
        #     phase = np.angle(raw)
        #     self.fft = np.abs(raw)
        #     self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
        #
        #     freqs = 60. * self.freqs
        #     idx = np.where((freqs > 50) & (freqs < 180))
        #
        #     pruned = self.fft[idx]
        #     phase = phase[idx]
        #
        #     pfreq = freqs[idx]
        #     self.freqs = pfreq
        #     self.fft = pruned
        #     idx2 = np.argmax(pruned)
        #
        #     t = (np.sin(phase[idx2]) + 1.) / 2.
        #     t = 0.9 * t + 0.1
        #     alpha = t
        #     beta = 1 - t
        #
        #     self.bpm = self.freqs[idx2]
        #     self.idx += 1
        #
        #     x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        #     r = alpha * self.frame_in[y:y + h, x:x + w, 0]
        #     g = alpha * \
        #         self.frame_in[y:y + h, x:x + w, 1] + \
        #         beta * self.gray[y:y + h, x:x + w]
        #     b = alpha * self.frame_in[y:y + h, x:x + w, 2]
        #     self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
        #                                                   g,
        #                                                   b])
        #     x1, y1, w1, h1 = self.face_rect
        #     self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
        #     col = (100, 255, 100)
        #     gap = (self.buffer_size - L) / self.fps
        #     # self.bpms.append(self.bpm)
        #     # self.ttimes.append(time.time())
        #     if gap:
        #         text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
        #     else:
        #         text = "(estimate: %0.1f bpm)" % (self.bpm)
        #     tsize = 1
        #     cv2.putText(self.frame_out, text,
        #                 (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)
