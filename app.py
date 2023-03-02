from kivymd.app import MDApp
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.label import MDLabel

from kivymd.uix.button import MDFillRoundFlatButton
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextFieldRect

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserIconView

from PIL import Image as PilImage

import os
import cv2
import torch
import numpy as np
import draw
import time
import PoseModule
import pickle
import detect
import analyse
import validify


framerate = 0

started, finished = False, False
bbox = (0, 0, 0, 0)
tracker = 0

pose = PoseModule.poseDetector()

f = open('sbd-model1.p', 'rb')
exerciseClassifier = pickle.load(f)
f.close()

# plateDetectionModel = torch.hub.load('./yolo/yolov5-master', 'custom', path='./yolo/yolov5-master/best.pt', source='local')  # local repo

plateDetectionModel = torch.hub.load('./yolov5', 'custom', path='./yolov5/best.pt', source='local')  # local repo


repTimes = []
fpsList = []
t, pTime, cTime = 0, 0, 0


x, y, prevY, prevX = 0, 0, 0, 0
velocity, xVelocity = 0, 0

topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

startY, startX, endY = 0, 0, 0

predicted, foundPlate, exercise, isDeadlift = False, False, None, False

stage, count, pauseStart, currentRepDuration, prevRepDuration = -1, 0, 0, 0, 0

d, v = [], []

xp, yp = 0, 0

temp = 0

averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint = 0, 0, 0, 0, 0

checkForm, save, showPose, recording = True, True, True, None

actual_x, actual_y = -1, -1

class MainWindow(Screen):
    pass
    # def on_enter(self, *args):
    #     self.button = MDFillRoundFlatButton(text='Select File')
    #     self.button.bind(on_press=self.show_file_chooser)
    #     self.add_widget(self.button)

    #     # Create a file chooser but hide it initially
    #     self.file_chooser = FileChooserIconView()
    #     self.file_chooser.bind(on_submit=self.file_selected)
    #     self.file_chooser.opacity = 0
    #     self.file_chooser.size_hint_y = None
    #     self.file_chooser.height = 0
    #     self.add_widget(self.file_chooser)

    # def show_file_chooser(self, *args):
    #     # Show the file chooser when the button is clicked
    #     self.file_chooser.opacity = 1
    #     self.file_chooser.size_hint_y = 1
    #     self.file_chooser.height = 200

    # def file_selected(self, fc, selection, touch):
    #     # Print the path of the selected file to the terminal
    #     print(selection[0])

    #     # Hide the file chooser after the user selects a file
    #     self.file_chooser.opacity = 0
    #     self.file_chooser.size_hint_y = None
    #     self.file_chooser.height = 0

class MyImage(Image):
    def on_touch_down(self, touch):
        global actual_x, actual_y
        if self.collide_point(*touch.pos):
            width, height = Window.size
            touch_x, touch_y = self.to_local(*touch.pos)

            # print(self.image_size)

            img_ratio = self.image_size[0]/ self.image_size[1]
            window_ratio = width/height

            ratio_ratio = window_ratio/img_ratio

            if ratio_ratio >= 1:

                actual_y = touch_y/height * self.image_size[1]
                # print("actual y: " + str(actual_y))
                # print("frac x: " + str(x_frac))

                image_size_on_window = width/ratio_ratio

                r1 = self.image_size[0]/image_size_on_window

                # print("size on window:" + str(image_size_on_window))

                actual_x = (touch_x - (width - image_size_on_window)/2)*r1

                actual_y = self.image_size[1] - actual_y

                # print("Actual position:", (actual_x, actual_y))

            else:
                actual_x = touch_x/width * self.image_size[0]
                # print("actual y: " + str(actual_y))
                # print("frac x: " + str(x_frac))

                image_size_on_window = height*ratio_ratio

                r1 = self.image_size[1]/image_size_on_window

                # print("size on window:" + str(image_size_on_window))

                actual_y = (touch_y - (height - image_size_on_window)/2)*r1

                actual_y = self.image_size[1] - actual_y

                # print("Actual position:", (actual_x, actual_y))
            

class SecondWindow(Screen):
    
    def extract_first_frame(self, video_filename, image_filename):
        # Open the video file
        cap = cv2.VideoCapture(video_filename)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return

        # Read the first frame of the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error reading first frame")
            cap.release()
            return

        # Save the first frame as an image file
        cv2.imwrite(image_filename, frame)

        # Release the video file
        cap.release()

        # print(f"First frame of {video_filename} saved as {image_filename}")


    def on_enter(self, *args):
        global bbox
        global foundPlate

        self.extract_first_frame(file, file + "first_frame.jpg")
        self.image = MyImage(source=file + "first_frame.jpg")
        self.add_widget(self.image, 100)
        os.remove(file + "first_frame.jpg")

        self.capture = cv2.VideoCapture(file)
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        
        size = (frame_width, frame_height)
        self.image.image_size = size

        ret, frame = self.capture.read()

        bbox, foundPlate = detect.findPlate(frame, plateDetectionModel)


        def set_bbox(dt):
            global actual_x, actual_y
            global bbox
            if actual_x < 0:
                return
            else:
                bbox = detect.findPlateAlt(frame, actual_x, actual_y)
                print(bbox)
                # bbox = (actual_x-100, actual_y-100, 200, 200)
                Clock.unschedule(set_bbox)

        if not bbox:
            Clock.schedule_interval(set_bbox, 0.1)


    def start(self):
        global file
        # self.image = MyImage()
        # self.add_widget(self.image, 100)

        self.exercise_button = MDFillRoundFlatButton(
            pos_hint = {"x": 0.1, "top": 0.95},
            size_hint = (None, None),
            size = (150, 50),
            text = "TBC",
            md_bg_color = "gray"
        )
        self.add_widget(self.exercise_button)

        self.rep_count_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.95},
            size_hint = (None, None),
            size = (50, 50),
            text = str(count),
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_count_button)

        self.rep_duration_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.15},
            size_hint = (None, None),
            size = (150, 50),
            text = "Rep duration:\n" + str(0) + "s",
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_duration_button)

        self.rep_duration_arrow = MDFillRoundFlatButton(
            pos_hint = {"x": 0.1, "top": 0.15},
            size_hint = (None, None),
            size = (150, 50),
            text = "-",
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_duration_arrow)

        if checkForm:
            self.form_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.55},
            size_hint = (None, None),
            size = (150, 50),
            text = "-",
            md_bg_color = "gray"
            )
            self.add_widget(self.form_button)

        self.capture = cv2.VideoCapture(file)
        framerate = self.capture.get(cv2.CAP_PROP_FPS)

        print(file)
        
        if framerate == 0:
            print("FILE NOT FOUND")
            return self
        
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        
        size = (frame_width, frame_height)
        self.image.image_size = size
        
        if save:
            global recording
            recording = cv2.VideoWriter('saved.mp4', cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)



        Clock.schedule_interval(self.load_video, 1/framerate)
        
        return self
        
    def update_rep_count_button(self):
        self.rep_count_button.text = "Rep count: " + str(count)

    def update_rep_duration_button(self):
        self.rep_duration_button.text = "Rep duration:\n" + str(round(currentRepDuration, 2)) + "s"

    def update_rep_duration_arrow(self):
        if currentRepDuration == 0:
            return
        self.rep_duration_arrow.text = "^" if currentRepDuration > prevRepDuration else "v"

    def rem_widgets(self):
        self.remove_widget(self.image)
        self.remove_widget(self.exercise_button)
        self.remove_widget(self.rep_count_button)
        self.remove_widget(self.rep_duration_button)
        self.remove_widget(self.rep_duration_arrow)
        if checkForm:
            self.remove_widget(self.form_button)

    def analyse(self):
        global framerate
        global started, finished
        global t, pTime, cTime
        global x, y, prevY, prevX
        global velocity, xVelocity
        global topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9
        global startY, startX, endY
        global predicted, foundPlate, exercise, isDeadlift
        global stage, count, pauseStart, currentRepDuration, prevRepDuration
        global xp, yp
        global repTimes
        global pose
    
        global averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint
        
        m = analyse.getAnalysis(startY, endY, repTimes, fpsList, framerate, bbox, isDeadlift, v, d, foundPlate, t)
        if not m:
            print("NO REPS")
            averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint = 0, 0, 0, 0, 0
        else:
            (averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint) = m

        framerate = 0
        started, finished = False, False
        t, pTime, cTime = 0, 0, 0
        x, y, prevY, prevX = 0, 0, 0, 0
        velocity, xVelocity = 0, 0
        topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        startY, startX, endY = 0, 0, 0
        predicted, foundPlate, exercise, isDeadlift = False, False, None, False
        stage, count, pauseStart, currentRepDuration, prevRepDuration = -1, 0, 0, 0, 0
        xp, yp = 0, 0
        repTimes = []
        pose = PoseModule.poseDetector()

        self.rem_widgets()

        if save:
            global recording
            recording.release()

        # ThirdWindow().set_average_rep_duration_button("New text for avgrepdur button")

        # set_average_rep_duration_button(m1)
        # App.get_running_app().root.ids.t1.text = "ppppp"
        # self.app.root.ThirdWindow.ids.t1.text = "ppppp"

    def load_video(self, *args):
        global file
        global framerate
        global started, finished
        global bbox
        global t, pTime, cTime
        global x, y, prevY, prevX
        global velocity, xVelocity
        global topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9
        global startY, startX, endY
        global predicted, foundPlate, exercise, isDeadlift
        global stage, count, pauseStart, currentRepDuration, prevRepDuration
        global imgCanvas
        global xp, yp
        global repTimes
        global tracker
        global pose
        global save, recording

        # print(repTimes)
        ret, frame = self.capture.read()
        if not ret:
            finished = True
            file = -1
            return False

        if file == -1:
            framerate = 0
            started, finished = False, False
            t, pTime, cTime = 0, 0, 0
            x, y, prevY, prevX = 0, 0, 0, 0
            velocity, xVelocity = 0, 0
            topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            startY, startX, endY = 0, 0, 0
            predicted, foundPlate, exercise, isDeadlift = False, False, None, False
            stage, count, pauseStart, currentRepDuration, prevRepDuration = -1, 0, 0, 0, 0
            xp, yp = 0, 0
            repTimes = []
            frame = None
            pose = PoseModule.poseDetector()

            self.rem_widgets()

            return False

        if not started:
            cap = cv2.VideoCapture(file)
            framerate = cap.get(cv2.CAP_PROP_FPS)


            # bbox, foundPlate = detect.findPlate(frame, plateDetectionModel)



            started = True

            height = frame.shape[0]
            width = frame.shape[1]
            imgCanvas = np.zeros((height,  width, 3), np.uint8)

            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, bbox)

        draw.drawBox(frame, bbox)

        t += 1
        cTime = time.time()
        if cTime - pTime != 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 30
        pTime = cTime
        fpsList.append(fps)


        # frame = cv2.putText(frame, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

        _, bbox = tracker.update(frame)

        x, y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        velocity = (prevY - y)*fps
        xVelocity = (prevX - x)*fps
        prevY = y
        prevX = y

        smoothedVelocity = (prevVelocity1 + prevVelocity2 + prevVelocity3 + prevVelocity4 + prevVelocity5 + prevVelocity6 + prevVelocity7 + prevVelocity8 + prevVelocity9 + velocity) * 0.1
        prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9, velocity

        topSpeed = max(topSpeed, abs(smoothedVelocity))

        
        if not predicted and (abs(velocity) > 1):
            predicted = True
            exercise = detect.predict(frame, exerciseClassifier, pose)
            # if exercise == "BENCH":
            #     while True:
            #         cv2.imshow("r", pose.findPoseEmpty(frame, draw=True))
            #         cv2.waitKey(1)
            self.exercise_button.text = exercise
            isDeadlift = exercise == "DEADLIFT"
            startY = 0 if isDeadlift else 1000
            endY = 1000 if isDeadlift else 0


        if predicted:
            if checkForm:
                    pose.findPose(frame, showPose)
            if exercise == "DEADLIFT":
                startY = max(startY, y)
                endY = min(endY, y)
                # for deadlift
                # -1 = start, 0 = at the bottom, 1 = going up, 2 = coming down
                # if stage == -1 and abs(smoothedVelocity) > 20:
                if stage == -1 and abs(y - startY) > 20:
                    stage = 1
                    print("going up")
                    repTimes.append(t)
                elif stage == 1 and smoothedVelocity < 0 and abs(y - startY) > 50:
                    stage = 2
                    print("coming down")
                    
                    if checkForm:
                        a = validify.lockedOut(frame, pose)
                        print(a)
                        self.form_button.text = "Good lockout" if a else "Bad lockout"

                    repTimes.append((repTimes.pop(), t))
                    avgFPS = np.mean(fpsList)
                    prevRepDuration = currentRepDuration
                    currentRepDuration = analyse.getAverageConcentricRepDuration([repTimes[-1]], avgFPS, isDeadlift, avgFPS/framerate)
                    self.update_rep_duration_button()
                    self.update_rep_duration_arrow()
                elif stage == 2 and abs(y - startY) < 50:
                    stage = 0
                    print("bottom")
                    count += 1
                    self.update_rep_count_button()
                    print("reps = " + str(count))
                    temp = repTimes.pop()
                    repTimes.append((temp[0], temp[1], t))
                    if checkForm:
                        self.form_button.text = "-"
                elif stage == 0 and smoothedVelocity > 0 and abs(y - startY) > 50:
                    stage = 1
                    print("going up")
                    repTimes.append(t)

            elif exercise == "SQUAT":
                # pose.findPose(img, False)
                startY = min(startY, y)
                endY = max(endY, y)
                # for squat
                # -1 = start, 0 = at the top, 1 = going down, 2 = coming up
                if stage == -1 and abs(y - startY) > 40:
                    startX = x
                    stage = 1
                    print("going down")
                    repTimes.append(t)
                elif stage == 1 and smoothedVelocity > 0 and abs(y - startY) > 50:
                    stage = 2
                    print("coming up")
                    if checkForm:
                        a = validify.goodDepth(frame, pose)
                        print(a)
                        self.form_button.text = "Good depth" if a else "Bad depth"
                    repTimes.append((repTimes.pop(), t))
                elif stage == 2 and abs(y - startY) < 20:
                    print("top")
                    stage = 0
                    count += 1
                    self.update_rep_count_button()
                    print("reps = " + str(count))
                    temp = repTimes.pop()
                    repTimes.append((temp[0], temp[1], t))
                    avgFPS = np.mean(fpsList)
                    prevRepDuration = currentRepDuration
                    currentRepDuration = analyse.getAverageConcentricRepDuration([repTimes[-1]], avgFPS, isDeadlift, avgFPS/framerate)
                    self.update_rep_duration_button()
                    self.update_rep_duration_arrow()
                    if checkForm:
                        self.form_button.text = "-"
                elif stage == 0 and smoothedVelocity < 0 and abs(y - startY) > 40 and abs(x - startX) < 30:
                    stage = 1
                    print("going down")
                    repTimes.append(t)

            else:
                startY = min(startY, y)
                endY = max(endY, y)
                # for bench
                # -1 = start, 0 = at the top, 1 = going down, 2 = coming up
                if stage == -1 and abs(y - startY) > 20:
                    startX = x
                    stage = 1
                    print("going down")
                    repTimes.append(t)
                elif stage == 1 and smoothedVelocity > 0 and abs(y - startY) > 50:
                    stage = 1.5
                    print("paused")
                    pauseStart = t
                elif stage == 1.5 and smoothedVelocity > 20:
                    stage = 2
                    print("coming up")
                    repTimes.append((repTimes.pop(), t))

                    if checkForm:
                        numOfPausedFrames = t - pauseStart
                        pausedTime = numOfPausedFrames/framerate
                        print(pausedTime)
                        if pausedTime >= 0.5:
                            print("GOOD PAUSE")
                            self.form_button.text = "Good pause"
                        else:
                            print("BAD PAUSE")
                            self.form_button.text = "Bad pause"
                    
                elif stage == 2 and abs(y - startY) < 20:
                    print("top")
                    stage = 0
                    count += 1
                    self.update_rep_count_button()
                    print("reps = " + str(count))
                    temp = repTimes.pop()
                    repTimes.append((temp[0], temp[1], t))
                    avgFPS = np.mean(fpsList)
                    prevRepDuration = currentRepDuration
                    currentRepDuration = analyse.getAverageConcentricRepDuration([repTimes[-1]], avgFPS, isDeadlift, avgFPS/framerate)
                    self.update_rep_duration_button()
                    self.update_rep_duration_arrow()
                    if checkForm:
                        self.form_button.text = "-"
                    
                elif stage == 0 and smoothedVelocity < 0 and abs(y - startY) > 20 and abs(x - startX) < 30:
                    stage = 1
                    print("going down")
                    repTimes.append(t)

        d.append(y)
        v.append(smoothedVelocity)

        frame, xp, yp = draw.drawPath(frame, bbox, smoothedVelocity, topSpeed, imgCanvas, xp, yp)

        if save:
                # overlay info
                recording.write(frame)


        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


class ThirdWindow(Screen):
    def on_enter(self, *args): # Executed when the ThirdWindow becomes the main window
        if stickingPoint == 0:
            self.ids.avgrepdur.text = "No reps detected"
            self.ids.avgconcrepdur.text = "No reps detected"
            self.ids.peakconcvel.text = "No reps detected"
            self.ids.avgconcvel.text = "No reps detected"
            self.ids.stickpnt.text = "No reps detected"
            return

        self.ids.avgrepdur.text = averageRepDuration
        self.ids.avgconcrepdur.text = averageConcentricRepDuration
        self.ids.peakconcvel.text = peakConcentricVelocity
        self.ids.avgconcvel.text = averageConcentricVelocity
        self.ids.stickpnt.text = stickingPoint

        self.ids.graph.source = "foo.png"

class WindowManager(ScreenManager):
    pass


kv = """

WindowManager:
    MainWindow:
    SecondWindow:
    ThirdWindow:

<MainWindow>:
    name: "main"
    
    MDLabel:
        text: "File name: "
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.95, "x": 0.1}

    MDTextFieldRect:
        multiline: False
        id: file
        hint_text:"Enter file name"
        height: "40dp"
        width: "10mm"
        pos_hint:{"top": 0.95, "x": 0.25}
        size_hint:0.5,None

    MDRaisedButton:
        text: "Submit"
        md_bg_color: app.theme_cls.primary_color
        size_hint_y: None
        height: "40dp"
        pos_hint: {"top": 0.95, "x": 0.8}
        on_release:
            app.set_file(file.text)
            app.set_switch_vars(checkform.active, save.active, showpose.active)
            app.root.current = "second"
            root.manager.transition.direction = "left"

    MDLabel:
        text: "Form check"
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.25, "x": 0.1}

    MDSwitch:
        id: checkform
        size_hint: None, None
        size: "48dp", "48dp"
        pos_hint: {"top": 0.15, "x": 0.1}

    MDLabel:
        text: "Save"
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.25, "x": 0.5}

    MDSwitch:
        id: save
        size_hint: None, None
        size: "48dp", "48dp"
        pos_hint: {"top": 0.15, "x": 0.5}

    MDLabel:
        text: "Show pose"
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.25, "x": 0.8}

    MDSwitch:
        id: showpose
        size_hint: None, None
        size: "48dp", "48dp"
        pos_hint: {"top": 0.15, "x": 0.8}


<SecondWindow>:
    name: "second"

    MDFillRoundFlatButton:
        id: b1
        text: "start"
        pos_hint: {"center_x": 0.5, "center_y": 0.95}
        size_hint: (None, None)
        size: (150, 50)
        mode: 0
        on_release:
            if self.mode == 0: \
            root.start(); \
            self.text = "back"; \
            self.mode = 1;
            else: \
            app.root.current = "main"; \
            root.manager.transition.direction = "right"; \
            app.set_file(-1); \
            self.text = "start"; \
            self.mode = 0;

    MDFillRoundFlatButton:
        text: "end"
        pos_hint: {"center_x": 0.5, "center_y": 0.05}
        size_hint: (None, None)
        size: (150, 50)
        on_release:
            root.ids.b1.text = "start"
            root.ids.b1.mode = 0
            root.analyse()
            app.set_file(-1)
            app.root.current = "third"
            root.manager.transition.direction = "left"


<ThirdWindow>:
    name: "third"

    MDFillRoundFlatButton:
        font_size: 20
        color: 0, 0, 0, 1
        background_color: 1, 1, 1, 1
        pos_hint: {"right": 0.2, "top": 0.95}
        size_hint: (None, None)
        size: (150, 50)
        text: "back"
        mode: 0
        on_release:
            app.root.current = "main"
            root.manager.transition.direction = "right"

    GridLayout:
        rows: 6

        MDLabel:
            text: 'Average rep duration'
            size_hint_x: 0.5
            halign: "center"

        MDLabel:
            id: avgrepdur
            text: 'Blank'
            size_hint_x: 0.2
            halign: "center"

        MDLabel:
            text: 'Average concentric rep duration'
            size_hint_x: 0.5
            halign: "center"

        MDLabel:
            id: avgconcrepdur
            text: 'Blank'
            size_hint_x: 0.2
            halign: "center"

        MDLabel:
            text: 'Peak concentric velocity'
            size_hint_x: 0.5
            halign: "center"

        MDLabel:
            id: peakconcvel
            text: 'Blank'
            size_hint_x: 0.2
            halign: "center"

        MDLabel:
            text: 'Average concentric velocity'
            size_hint_x: 0.5
            halign: "center"

        MDLabel:
            id: avgconcvel
            text: 'Blank'
            size_hint_x: 0.2
            halign: "center"

        MDLabel:
            text: 'Sticking Point'
            size_hint_x: 0.2
            halign: "center"
        
        MDLabel:
            id: stickpnt
            text: 'Blank'
            size_hint_x: 0.2
            halign: "center"

        Image:
            id: graph 

"""


class MainApp(MDApp):
    def build(self):
        return Builder.load_string(kv)

    def on_start(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Indigo"
        self.theme_cls.accent_palette = "Pink"

    def set_file(self, f):
        global file
        if f == -1:
            file = f
            return
        file = "vids\\" + f

    def set_switch_vars(self, a, b, c):
        global checkForm, save, showPose
        checkForm, save, showPose = a, b, c



if __name__ == "__main__":
    MainApp().run()
