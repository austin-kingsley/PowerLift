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
from kivy.uix.popup import Popup
from kivymd.uix.button import MDIconButton

from kivy.config import Config
Config.set('kivy','window_icon','icon.ico')

from PIL import Image as PilImage

import os
import shutil
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

import sklearn
f = open('sbd-model1-best.p', 'rb')
exerciseClassifier = pickle.load(f)
f.close()

# plateDetectionModel = torch.hub.load('./yolo/yolov5-master', 'custom', path='./yolo/yolov5-master/best.pt', source='local')  # local repo
import IPython
import psutil
import torchvision
import yaml
import scipy
import tqdm
import seaborn
import PIL
plateDetectionModel = torch.hub.load('./yolov5', 'custom', path='./yolov5/best.pt', source='local')  # local repo


repTimes = []
fpsList = []
t, pTime, cTime = 0, 0, 0

x, y, prevY, prevX = 0, 0, 0, 0
velocity, xVelocity = 0, 0

topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

startY, startX, endY = 0, 0, 0

predicted, foundPlate, exercise, isDeadlift = False, False, None, False

stage, stageText, formText, count, pauseStart, currentRepDuration, prevRepDuration = -1, "", "", 0, 0, 0, 0

d, v = [], []

xp, yp = 0, 0

temp = 0

averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint = 0, 0, 0, 0, 0

checkForm, save, showPose, recording = True, True, True, None

actual_x, actual_y = -1, -1

rom, peak_rep_vel, avg_rep_vel = 0, 0, 0

overlay = True

temp_file = 0

# information popup for MainWindow
class MyPopup1(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        content = BoxLayout(orientation='vertical', padding=10, spacing=0)

        self.text = MDLabel(text="""Welcome to PowerLyft, the ultimate tool for powerlifters who want to improve their lifts! With this app, you can input a recording of your lifts and get detailed feedback on your form, velocity, and more.
                                \nSelect a video to analyse and toggle the switches at the bottom of the screen.
                                \n'Form check' lets you know if you're performing each rep with good form.
                                \n'Save' lets you save the analysed video, including the bar path and overlay of your velocity info.
                                \n'Show pose' displays the skeletal pose of your lift throughout.
                                \nOnce you submit a video, the next screen will play it and classify the exercise into squat, bench press or deadlift, as well as automatically detect the end of the barbell, trace the bar path, and display detailed velocity metrics.
                                \nEnjoy!""")
        content.add_widget(self.text)
        self.content = content
        self.size_hint = (0.6, 0.95)
        self.title = 'Application Info'

# information popup for SecondWindow
class MyPopup2(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        content = BoxLayout(orientation='vertical', padding=10, spacing=0)
        self.text = MDLabel(text="""On this screen you can see an overview of the analysis, including a graph showing the velocity and displacement of the bar against time. With just a single click, you can save this graph to your Pictures directory.
                                \nParticular metrics relating to your lift can also be seen here, which you should note down to keep track of progress over time.
                                \nCalandar view coming soon!
                                \nIf a sticking point has been identified it has been used to determine corrective exercises you should implement into your routine.
                                \nClick 'Back' to analyse another video.""")
        content.add_widget(self.text)
        self.content = content
        self.size_hint = (0.6, 0.75)
        self.title = 'Application Info'

# startup page
class MainWindow(Screen):
    # toggle visibility of overlay switch
    def toggle_overlay_show(self):
        self.ids.overlay.opacity = 1 - self.ids.overlay.opacity
        self.ids.overlaytext.opacity = 1 - self.ids.overlaytext.opacity
    
    # toggle overlay bool
    def toggle_overlay(self):
        global overlay
        overlay = not overlay

    # on entering MainWindow
    def on_enter(self, *args):
        # Create a file chooser but hide it initially
        p = os.path.join(os.path.expanduser('~'), 'Videos')
        self.file_chooser = FileChooserIconView(path=p, pos_hint={"top": 0.9}, filters = ['*.mp4', '*.MOV'])
        self.file_chooser.bind(on_submit=self.file_selected)
        self.file_chooser.opacity = 0
        self.file_chooser.size_hint_y = 0.55
        self.file_chooser.height = 200
        self.add_widget(self.file_chooser)

        # show information popup
        def show_popup(instance):
            popup = MyPopup1()
            close_button = MDFillRoundFlatButton(text="Close", size_hint=(0.2, 0.1), pos_hint={"right": 0.99})
            close_button.bind(on_press=popup.dismiss)
            popup.content.add_widget(close_button)
            popup.open()

        button = MDIconButton(icon="information", pos_hint={"top": 0.98, "right": 0.99})
        button.bind(on_press=show_popup)
        self.add_widget(button)

    def show_file_chooser(self, *args):
        # Show the file chooser when the button is clicked
        if self.file_chooser.opacity == 0:
            self.file_chooser.opacity = 1
            self.file_chooser.size_hint_y = 0.55
            self.file_chooser.height = 200
        else: # hide when clicked again
            self.file_chooser.opacity = 0


    def file_selected(self, fc, selection, touch):
        # Set the filename label
        self.ids.filename.text = selection[0]
        self.file_chooser.opacity = 0
        self.ids.submit.opacity = 1

class MyImage(Image):
    # upon a user click, find its position on the image, set actual_x and actual_y
    def on_touch_down(self, touch):
        global actual_x, actual_y
        if self.collide_point(*touch.pos):
            width, height = Window.size
            touch_x, touch_y = self.to_local(*touch.pos)

            img_ratio = self.image_size[0]/ self.image_size[1]
            window_ratio = width/height

            ratio_ratio = window_ratio/img_ratio

            if ratio_ratio >= 1:

                actual_y = touch_y/height * self.image_size[1]

                image_size_on_window = width/ratio_ratio

                r1 = self.image_size[0]/image_size_on_window

                actual_x = (touch_x - (width - image_size_on_window)/2)*r1

                actual_y = self.image_size[1] - actual_y

            else:
                actual_x = touch_x/width * self.image_size[0]

                image_size_on_window = height*ratio_ratio

                r1 = self.image_size[1]/image_size_on_window

                actual_y = (touch_y - (height - image_size_on_window)/2)*r1

                actual_y = self.image_size[1] - actual_y

# page playing video
class SecondWindow(Screen):
    # get the first frame of an input video, save to image file
    def extract_first_frame(self, video_filename, image_filename):
        # open the video file
        cap = cv2.VideoCapture(video_filename)

        # check if the video file was opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return

        # read the first frame of the video
        ret, frame = cap.read()

        # check if the frame was read successfully
        if not ret:
            print("Error reading first frame")
            cap.release()
            return

        # save the first frame as an image file
        cv2.imwrite(image_filename, frame)

        # release the video file
        cap.release()

    # on entering SecondWindow
    def on_enter(self, *args):
        global bbox
        global foundPlate

        self.ids.endb.opacity = 0

        # set image as first frame of chosen video
        self.extract_first_frame(file, file + "first_frame.jpg")
        self.image = MyImage(source=file + "first_frame.jpg")
        self.add_widget(self.image, 100)
        os.remove(file + "first_frame.jpg") # remove the temp image file

        # open video file and get ready to capture
        self.capture = cv2.VideoCapture(file)
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        size = (frame_width, frame_height)
        self.image.image_size = size
        ret, frame = self.capture.read()

        # identify the end of the weight plate based on first frame
        bbox, foundPlate = detect.findPlate(frame, plateDetectionModel)

        # set the bounding box according to output of YOLO object detector, or redundant system if YOLO fails
        def set_bbox(dt):
            global actual_x, actual_y
            global bbox
            if actual_x < 0:
                self.ids.startb.opacity = 0
                self.warning.text = "No weight plate detected. Please click the end of the barbell"
                return
            else:
                bbox = detect.findPlateAlt(frame, actual_x, actual_y)
                print(bbox)
                self.remove_widget(self.warning)
                self.ids.startb.opacity = 1
                Clock.unschedule(set_bbox)

        # repeatedly prompt user to click on the screen if YOLO fails
        if not bbox:
            self.warning = MDLabel(size_hint = (0.2, 1), pos_hint = {"top": 0.97, "x": 0.02}, opacity = 1)
            self.add_widget(self.warning)

            Clock.schedule_interval(set_bbox, 0.1)

    # when the user decides to start the analysis
    def start(self):
        global file

        # used to display exercise being performed
        self.exercise_button = MDFillRoundFlatButton(
            pos_hint = {"x": 0.1, "top": 0.95},
            size_hint = (None, None),
            size = (150, 50),
            text = "TBC",
            md_bg_color = "gray"
        )
        self.add_widget(self.exercise_button)

        # used to display number of completed repetitions
        self.rep_count_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.95},
            size_hint = (None, None),
            size = (50, 50),
            text = "Rep count: 0",
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_count_button)

        # used to display average velocity metric
        self.avg_rep_velocity_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.45},
            size_hint = (None, None),
            size = (150, 50),
            text = "Avg concentric velocity:\n" + str(0) + "m/s",
            md_bg_color = "gray"
        )
        self.add_widget(self.avg_rep_velocity_button)

        # used to display peak velocity metric
        self.peak_rep_velocity_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.35},
            size_hint = (None, None),
            size = (150, 50),
            text = "Peak concentric velocity:\n" + str(0) + "m/s",
            md_bg_color = "gray"
        )
        self.add_widget(self.peak_rep_velocity_button)

        # used to display range of motion metric
        self.rom_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.25},
            size_hint = (None, None),
            size = (150, 50),
            text = "ROM: " + str(0) + "m",
            md_bg_color = "gray"
        )
        self.add_widget(self.rom_button)

        # used to display current rep duration metric
        self.rep_duration_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.15},
            size_hint = (None, None),
            size = (150, 50),
            text = "Conc rep duration:\n" + str(0) + "s",
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_duration_button)

        # used to display the stage of the rep
        self.rep_stage_button = MDFillRoundFlatButton(
            pos_hint = {"x": 0.1, "top": 0.15},
            size_hint = (None, None),
            size = (150, 50),
            text = "Start",
            md_bg_color = "gray"
        )
        self.add_widget(self.rep_stage_button)

        # used to display result of form checker
        if checkForm:
            self.form_button = MDFillRoundFlatButton(
            pos_hint = {"right": 0.95, "top": 0.55},
            size_hint = (None, None),
            size = (150, 50),
            text = "-",
            md_bg_color = "gray"
            )
            self.add_widget(self.form_button)

        # get framerate of video file
        self.capture = cv2.VideoCapture(file)
        framerate = self.capture.get(cv2.CAP_PROP_FPS)

        if framerate == 0:
            print("FILE NOT FOUND")
            return self
        
        # get dimensions of video
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        size = (frame_width, frame_height)
        self.image.image_size = size
        
        # initialise video recorder object if the user would like to save the analysis
        if save:
            global recording
            recording = cv2.VideoWriter(file.split('.')[:-1][0]+"-saved.mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)

        # play each frame according to framerate of video file
        Clock.schedule_interval(self.load_video, 1/framerate)
        
        return self
        
    def update_rep_count_button(self):
        self.rep_count_button.text = "Rep count: " + str(count)

    def update_rep_duration_button(self):
        self.rep_duration_button.text = "Conc rep duration:\n" + str(round(currentRepDuration, 2)) + "s"

    def update_rep_velocity_and_rom_buttons(self):
        global rom, peak_rep_vel, avg_rep_vel

        # find the range of motion in metres
        pixelsPerRep = startY - endY
        pixelsPerWeightPlate = abs(bbox[3])
        metersPerWeightPlate = 0.45
        metersPerPixel = metersPerWeightPlate/pixelsPerWeightPlate
        rom = abs(pixelsPerRep*metersPerPixel)

        avg_rep_vel = abs(rom/currentRepDuration) # find average rep velocity
        new_v = [metersPerWeightPlate*x/pixelsPerWeightPlate for x in v] # find the velocity at each frame in metres per second

        # find peak rep velocity
        peak_rep_vel = analyse.getPeakConcentricVelocity([repTimes[-1]], new_v, isDeadlift, np.mean(fpsList)/framerate)

        # update buttons
        self.avg_rep_velocity_button.text = "Avg concentric velocity:\n" + str(round(avg_rep_vel, 2)) + "m/s"
        self.peak_rep_velocity_button.text = "Peak concentric velocity:\n" + str(round(peak_rep_vel, 2)) + "m/s"
        self.rom_button.text = "ROM: " + str(round(rom, 2)) + "m"

    def update_rep_stage_button(self, stage):
        self.rep_stage_button.text = stage

    # remove widgets at the end of the video
    def rem_widgets(self):
        self.remove_widget(self.image)
        self.remove_widget(self.exercise_button)
        self.remove_widget(self.rep_count_button)
        self.remove_widget(self.rep_duration_button)
        self.remove_widget(self.avg_rep_velocity_button)
        self.remove_widget(self.peak_rep_velocity_button)
        self.remove_widget(self.rom_button)
        self.remove_widget(self.rep_stage_button)
        if checkForm:
            self.remove_widget(self.form_button)

    # overlay metrics onto the recording to be saved, if the user toggled the overlay switch
    def overlay(self, frame):
        if not overlay:
            return frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (0, 255, 0)
        thickness = 3
        height = frame.shape[0]

        frame = cv2.putText(frame, exercise, (0, 100), font, fontScale, color, thickness, cv2.LINE_AA, False) if exercise else frame
        frame = cv2.putText(frame, stageText, (0, height-100), font, fontScale, color, thickness, cv2.LINE_AA, False)   
        frame = cv2.putText(frame, "Rep count: " + str(count), (0, 200), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, "Conc rep duration: " + str(round(currentRepDuration, 2)) + "s", (0, height-200), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, "ROM: " + str(round(rom, 2)) + "m", (0, height-300), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, "Peak velocity: " + str(round(peak_rep_vel, 2)) + "m/s", (0, height-400), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, "Avg velocity: " + str(round(avg_rep_vel, 2)) + "m/s", (0, height-500), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, formText, (0, height-600), font, fontScale, color, thickness, cv2.LINE_AA, False) if checkForm else frame
            
        return frame

    # analyse the lift once the video is complete
    def analyse(self):
        global fpsList, framerate
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
        global v, d
        global averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint

        # get name of file
        f = temp_file.split('\\')[-1].split('.')[0]
        print(f)
        
        # perform analysis and set vars
        m = analyse.getAnalysis(startY, endY, repTimes, fpsList, framerate, bbox, isDeadlift, v, d, foundPlate, t, f)
        if not m:
            print("NO REPS")
            averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint = 0, 0, 0, 0, 0
        else:
            (averageRepDuration, averageConcentricRepDuration, peakConcentricVelocity, averageConcentricVelocity, stickingPoint) = m

        # reset global vars for next analysis
        fpsList, framerate = [], 0
        started, finished = False, False
        t, pTime, cTime = 0, 0, 0
        x, y, prevY, prevX = 0, 0, 0, 0
        velocity, xVelocity = 0, 0
        topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        startY, startX, endY = 0, 0, 0
        predicted, foundPlate, isDeadlift = False, False, False
        stage, count, pauseStart, currentRepDuration, prevRepDuration = -1, 0, 0, 0, 0
        xp, yp = 0, 0
        v, d = [], []
        repTimes = []
        pose = PoseModule.poseDetector()

        self.rem_widgets()

        # save the recording
        if save:
            global recording
            recording.release()

    # play a frame of the video,
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
        global stage, stageText, formText, count, pauseStart, currentRepDuration, prevRepDuration
        global imgCanvas
        global xp, yp
        global repTimes
        global tracker
        global pose
        global save, recording

        # if video is over
        ret, frame = self.capture.read()
        if not ret:
            finished = True
            file = -1
            return False

        # if user has ended video early
        if file == -1:
            framerate = 0
            started, finished = False, False
            t, pTime, cTime = 0, 0, 0
            x, y, prevY, prevX = 0, 0, 0, 0
            velocity, xVelocity = 0, 0
            topSpeed, smoothedVelocity, prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            startY, startX, endY = 0, 0, 0
            predicted, foundPlate, isDeadlift = False, False, False
            stage, count, pauseStart, currentRepDuration, prevRepDuration = -1, 0, 0, 0, 0
            xp, yp = 0, 0
            repTimes = []
            frame = None
            pose = PoseModule.poseDetector()

            self.rem_widgets()

            return False

        # initialise video capture
        if not started:
            cap = cv2.VideoCapture(file)
            framerate = cap.get(cv2.CAP_PROP_FPS)
            started = True

            # set empty canvas to trace bar path onto
            height = frame.shape[0]
            width = frame.shape[1]
            imgCanvas = np.zeros((height,  width, 3), np.uint8)

            # initialise tracker on starting position of the detected weight plate
            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, bbox)

        # draw.drawBox(frame, bbox) # draw a box around the weight plate

        t += 1 # keep track of number of frames read

        # depricated
        cTime = time.time()
        if cTime - pTime != 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 30
        pTime = cTime
        fpsList.append(framerate)
        # frame = cv2.putText(frame, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

        _, bbox = tracker.update(frame) # update tracker on every frame

        # bbox = (x-value of bottom-left corner, x-value of bottom-left corner, width, height)
        x, y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2

        # get velocity of bbox in pixels per second
        velocity = (prevY - y)*framerate
        xVelocity = (prevX - x)*framerate
        prevY = y
        prevX = x

        # get moving average
        smoothedVelocity = (prevVelocity1 + prevVelocity2 + prevVelocity3 + prevVelocity4 + prevVelocity5 + prevVelocity6 + prevVelocity7 + prevVelocity8 + prevVelocity9 + velocity) * 0.1
        prevVelocity1, prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9 = prevVelocity2, prevVelocity3, prevVelocity4, prevVelocity5, prevVelocity6, prevVelocity7, prevVelocity8, prevVelocity9, velocity
        # smoothedVelocity = velocity

        # when the bar starts moving, call exercise classifier
        if not predicted and (abs(velocity) > 1) and t > 1:
            predicted = True
            exercise = detect.predict(frame, exerciseClassifier, pose)
    
            self.exercise_button.text = exercise
            isDeadlift = exercise == "Deadlift"
            startY = 0 if isDeadlift else 1000
            endY = 1000 if isDeadlift else 0

        if predicted:
            if checkForm or showPose:
                    frame = pose.findPose(frame, showPose) # update the pose

            # for the deadlift
            if exercise == "Deadlift":
                startY = max(startY, y)
                endY = min(endY, y)
                # -1 = start, 0 = at the bottom, 1 = going up, 2 = coming down
                # go to stage 1 if at stage -1 and bar deviates 50 pixels vertically
                if stage == -1 and abs(y - startY) > 20: # 20 -> 50
                    stage = 1
                    print("going up")
                    self.update_rep_stage_button("Going up")
                    stageText = "Going up"
                    repTimes.append(t)

                # go to stage 2 if at stage 1, bar moves down, and bar is far from starting y-position
                elif stage == 1 and velocity < 0 and abs(y - startY) > 100:
                    stage = 2
                    print("coming down")
                    self.update_rep_stage_button("Coming down")
                    stageText = "Coming down"
                    
                    if checkForm: # check if lockout is sufficient
                        a = validify.lockedOut(frame, pose)
                        print(a)
                        self.form_button.text = "Good lockout" if a else "Bad lockout"
                        formText = "Good lockout" if a else "Bad lockout"

                    repTimes.append((repTimes.pop(), t))
                    avgFPS = np.mean(fpsList)
                    prevRepDuration = currentRepDuration
                    currentRepDuration = analyse.getAverageConcentricRepDuration([repTimes[-1]], avgFPS, isDeadlift, avgFPS/framerate)
                    self.update_rep_duration_button()
                    self.update_rep_velocity_and_rom_buttons()
                    
                # go to stage 0 if at stage 2 and bar nears starting y-position
                elif stage == 2 and abs(y - startY) < 50:
                    stage = 0
                    print("bottom")
                    count += 1
                    self.update_rep_count_button()
                    self.update_rep_stage_button("At bottom")
                    stageText = "At bottom"
                    print("reps = " + str(count))
                    temp = repTimes.pop()
                    repTimes.append((temp[0], temp[1], t))
                    if checkForm:
                        self.form_button.text = "-"
                
                # go to stage 1 if at stage 0 and bar deviates 50 pixels vertically
                elif stage == 0 and velocity > 0 and abs(y - startY) > 50:
                    stage = 1
                    print("going up")
                    self.update_rep_stage_button("Going up")
                    stageText = "Going up"
                    repTimes.append(t)

            # for the squat
            elif exercise == "Squat":
                startY = min(startY, y)
                endY = max(endY, y)
                # -1 = start, 0 = at the top, 1 = going down, 2 = coming up
                # go to stage 1 if at stage -1 and bar deviates 40 pixels vertically
                if stage == -1 and abs(y - startY) > 20:
                    startX = x
                    stage = 1
                    print("going down")
                    self.update_rep_stage_button("Going down")
                    stageText = "Going down"
                    repTimes.append(t)
                
                # go to stage 2 if at stage 1, bar moves up, and bar is far from starting y-position
                elif stage == 1 and velocity > 0 and abs(y - startY) > 100:
                    stage = 2
                    print("coming up")
                    self.update_rep_stage_button("Coming up")
                    stageText = "Coming up"

                    if checkForm: # check if depth is sufficient
                        a = validify.goodDepth(frame, pose)
                        print(a)
                        self.form_button.text = "Good depth" if a else "Bad depth"
                        formText = "Good depth" if a else "Bad depth"
                    repTimes.append((repTimes.pop(), t))

                # go to stage 0 if at stage 2, bar moves up, and bar is near starting y-position
                elif stage == 2 and abs(y - startY) < 20: # 20 -> 35
                    print("top")
                    self.update_rep_stage_button("At top")
                    stageText = "At top"
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
                    self.update_rep_velocity_and_rom_buttons()
                    if checkForm:
                        self.form_button.text = "-"

                # go to stage 1 if at stage 0, bar moves down, bar is far from starting y-position, and close to starting x-position
                elif stage == 0 and velocity < 0 and abs(y - startY) > 40 and abs(x - startX) < 30:
                    stage = 1
                    print("going down")
                    self.update_rep_stage_button("Going down")
                    stageText = "Going down"
                    repTimes.append(t)

            # for the bench press
            else:
                startY = min(startY, y)
                endY = max(endY, y)
                # -1 = start, 0 = at the top, 1 = going down, 1.5 = paused, 2 = coming up
                # go to stage 1 if at stage -1 and bar deviates 20 pixels vertically
                if stage == -1 and abs(y - startY) > 20:
                    startX = x
                    stage = 1
                    print("going down")
                    self.update_rep_stage_button("Going down")
                    stageText = "Going down"
                    repTimes.append(t)

                # go to stage 1.5 if at stage 1, bar stops moving down, and bar is far from starting y-position
                elif stage == 1 and smoothedVelocity >= 0 and abs(y - startY) > 100:
                    stage = 1.5
                    print("paused")
                    self.update_rep_stage_button("Paused")
                    stageText = "Paused"
                    pauseStart = t

                # go to stage 2 if at stage 1.5 and bar quickly moves up
                elif stage == 1.5 and velocity > 10:
                    stage = 2
                    print("coming up")
                    self.update_rep_stage_button("Coming up")
                    stageText = "Coming up"
                    repTimes.append((repTimes.pop(), t))

                    if checkForm: # check if pause is sufficient
                        if validify.goodPause(t, pauseStart, framerate):
                            print("GOOD PAUSE")
                            self.form_button.text = "Good pause"
                            formText = "Good pause"
                        else:
                            print("BAD PAUSE")
                            self.form_button.text = "Bad pause"
                            formText = "Bad pause"

                # go to stage 0 if at stage 2, and bar is near starting y-position
                elif stage == 2 and abs(y - startY) < 10: # 20 -> 30
                    print("top")
                    self.update_rep_stage_button("At top")
                    stageText = "At top"
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
                    self.update_rep_velocity_and_rom_buttons()
                    if checkForm:
                        self.form_button.text = "-"
                
                # go to stage 1 if at stage 0, bar moves down, bar is far from starting y-position, and close to starting x-position
                elif stage == 0 and velocity < 0 and abs(y - startY) > 10 and abs(x - startX) < 30:
                    stage = 1
                    print("going down")
                    self.update_rep_stage_button("Going down")
                    stageText = "Going down"
                    repTimes.append(t)

        d.append(y)
        v.append(smoothedVelocity)

        frame, xp, yp = draw.drawPath(frame, bbox, smoothedVelocity, imgCanvas, xp, yp) # trace the bar path on the current frame

        if save: # write to recording to be saved, with overlay
            recording.write(self.overlay(frame.copy()))


        # display the frame on the image widget
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

# page showing summary of analysis
class ThirdWindow(Screen):

    # save the graph to the user's Pictures directory
    def save_graph(self, _):
        f = temp_file.split('\\')[-1].split('.')[0] + "-graph.png"
        # p = os.path.join(os.path.expanduser('~'), 'Pictures', f)
        # self.graph.export_to_png(p)
        shutil.copyfile("foo.png", os.path.join(os.path.expanduser('~'), 'Pictures', f))
        
        self.saved = MDLabel(text = "Saved to Pictures!", pos_hint = {"x": 0.135, "top": 0.25}, size_hint = (1, None))
        self.add_widget(self.saved)

        def remove_label(dt):
            self.remove_widget(self.saved)
        Clock.schedule_interval(remove_label, 5)
        
    # on entering ThirdWindow
    def on_enter(self, *args):

        # show information popup
        def show_popup(instance):
            popup = MyPopup2()
            close_button = MDFillRoundFlatButton(text="Close", size_hint=(0.2, 0.1), pos_hint={"right": 0.99})
            close_button.bind(on_press=popup.dismiss)
            popup.content.add_widget(close_button)
            popup.open()

        button = MDIconButton(icon="information", pos_hint={"top": 0.98, "right": 0.99})
        button.bind(on_press=show_popup)
        self.add_widget(button)
        
        # if no reps have been completed
        if stickingPoint == 0:
            self.ids.avgrepdur.text = "No reps detected"
            self.ids.avgconcrepdur.text = "No reps detected"
            self.ids.peakconcvel.text = "No reps detected"
            self.ids.avgconcvel.text = "No reps detected"
            self.ids.stickpnt.text = "No reps detected"
            self.ids.recommendation.opacity = 0
            return

        # update buttons to display analysis
        self.ids.avgrepdur.text = averageRepDuration
        self.ids.avgconcrepdur.text = averageConcentricRepDuration
        self.ids.peakconcvel.text = peakConcentricVelocity
        self.ids.avgconcvel.text = averageConcentricVelocity
        self.ids.stickpnt.text = stickingPoint
        self.ids.recommendation.opacity = int(not stickingPoint == "None")

        # suggest exercises based on sticking point
        match exercise:
            case "Squat":
                if stickingPoint == "Top":
                    self.ids.recommendation.text = "Your main weakness is your lower back. Recommended exercises: Good mornings, Box squats"
                else:
                    self.ids.recommendation.text = "Your main weakness is your quads. Recommended exercises: Paused squats, Leg press"
            case "Bench":
                if stickingPoint == "Top":
                    self.ids.recommendation.text = "Your main weakness is your triceps. Recommended exercises: Close grip bench press, Floor press"
                else:
                    self.ids.recommendation.text = "Your main weakness is your chest. Recommended exercises: Long pause bench press, Dumbbell bench press"
            case "Deadlift":
                if stickingPoint == "Top":
                    self.ids.recommendation.text = "Your main weakness is your lower back. Recommended exercises: Box pulls, Good mornings"
                else:
                    self.ids.recommendation.text = "Your main weakness is your legs. Recommended exercises: Deficit deadlift, Wide grip deadlift"

        # set the graph image
        def set_graph(dt):
            f = temp_file.split('\\')[-1].split('.')[0]
            self.graph = Image(pos_hint = {"x": 0.025}, source = f + '.png', size_hint_x = 0.38)
            self.add_widget(self.graph)
            os.remove(f + '.png')

            self.savegraph = MDRaisedButton(text = "Save", pos_hint = {"x": 0.175, "top": 0.25}, on_release = self.save_graph)
            self.add_widget(self.savegraph)
            
            Clock.unschedule(set_graph)

        Clock.schedule_interval(set_graph, 0.1)

    # remove graph if the widget exists
    def remove_graph(self):
        if stickingPoint:
            self.remove_widget(self.graph)

class WindowManager(ScreenManager):
    pass

# kivy langauge file:
kv = """

WindowManager:
    MainWindow:
    SecondWindow:
    ThirdWindow:

<MainWindow>:
    name: "main"

    MDFillRoundFlatButton:
        text: "Select File"
        md_bg_color: app.theme_cls.primary_color
        size_hint_y: None
        height: "40dp"
        pos_hint: {"top": 0.97, "x": 0.1}
        on_release:
            root.show_file_chooser()

    MDLabel:
        id: filename
        text: ""
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.97, "x": 0.3}

    MDFillRoundFlatButton:
        id: submit
        text: "Submit"
        opacity: 0
        md_bg_color: app.theme_cls.primary_color
        size_hint_y: None
        height: "40dp"
        pos_hint: {"top": 0.97, "right": 0.9}
        on_release:
            opacity: 0
            app.set_file(filename.text)
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
        pos_hint: {"top": 0.25, "x": 0.45}

    MDSwitch:
        id: save
        size_hint: None, None
        size: "48dp", "48dp"
        pos_hint: {"top": 0.15, "x": 0.45}
        on_active:
            root.toggle_overlay_show()

    MDLabel:
        id: overlaytext
        text: "Overlay"
        size_hint: 1, None
        height: "40dp"
        pos_hint: {"top": 0.08, "x": 0.35}
        opacity: 0

    MDSwitch:
        id: overlay
        size_hint: None, None
        size: "48dp", "48dp"
        pos_hint: {"top": 0.08, "x": 0.45}
        active: True
        opacity: 0
        on_active:
            root.toggle_overlay()


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
        id: startb
        text: "start"
        pos_hint: {"center_x": 0.5, "center_y": 0.95}
        size_hint: (None, None)
        size: (150, 50)
        mode: 0
        on_release:
            if self.mode == 0: \
            root.start(); \
            self.text = "Back"; \
            self.mode = 1;
            else: \
            app.root.current = "main"; \
            root.manager.transition.direction = "right"; \
            app.set_file(-1); \
            self.text = "Start"; \
            self.mode = 0;
            root.ids.endb.opacity = 1

    MDFillRoundFlatButton:
        id: endb
        text: "End"
        pos_hint: {"center_x": 0.5, "center_y": 0.05}
        size_hint: (None, None)
        size: (150, 50)
        opacity: 0
        on_release:
            root.ids.startb.text = "Start"
            root.ids.startb.mode = 0
            root.analyse()
            app.root.current = "third"
            root.manager.transition.direction = "left"
            app.set_file(-1)


<ThirdWindow>:
    name: "third"

    MDFillRoundFlatButton:
        font_size: 20
        color: 0, 0, 0, 1
        background_color: 1, 1, 1, 1
        pos_hint: {"right": 0.2, "top": 0.95}
        size_hint: (None, None)
        size: (150, 50)
        text: "Back"
        mode: 0
        on_release:
            root.remove_graph()
            app.root.current = "main"
            root.manager.transition.direction = "right"

    GridLayout:
        rows: 6
        padding: 20

        MDLabel:
            text: 'Average rep duration'
            size_hint_x: 0.5
            halign: "right"

        MDLabel:
            id: avgrepdur
            text: ''
            size_hint_x: 0.2
            halign: "right"

        MDLabel:
            text: 'Average concentric rep duration'
            size_hint_x: 0.5
            halign: "right"

        MDLabel:
            id: avgconcrepdur
            text: ''
            size_hint_x: 0.2
            halign: "right"

        MDLabel:
            text: 'Peak concentric velocity'
            size_hint_x: 0.5
            halign: "right"

        MDLabel:
            id: peakconcvel
            text: ''
            size_hint_x: 0.2
            halign: "right"

        MDLabel:
            text: 'Average concentric velocity'
            size_hint_x: 0.5
            halign: "right"

        MDLabel:
            id: avgconcvel
            text: ''
            size_hint_x: 0.2
            halign: "right"

        MDLabel:
            text: 'Sticking Point'
            size_hint_x: 0.2
            halign: "right"
        
        MDLabel:
            id: stickpnt
            text: ''
            size_hint_x: 0.2
            halign: "right"

        MDLabel:
            text: ''
            size_hint_x: 0.2

        MDLabel:
            text: ''
            size_hint_x: 0.2

    MDFillRoundFlatButton:
        id: recommendation
        font_size: 15
        text: ''
        pos_hint: {"x": 0, "top": 0.1}
        size_hint_x: 1
        halign: "center"
        md_bg_color: [0,0,0,0]

"""

class MainApp(MDApp):
    # build the app
    def build(self):
        icon = 'icon.ico'
        return Builder.load_string(kv)

    # on starting the app
    def on_start(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Indigo"
        self.theme_cls.accent_palette = "Pink"

    # set the file name to either -1 or actual file name, based on state of video
    def set_file(self, f):
        global file
        global temp_file
        if f == -1:
            file = f
            return
        file = f
        temp_file = file

    # set vars based on state of switches in starting screen
    def set_switch_vars(self, a, b, c):
        global checkForm, save, showPose
        checkForm, save, showPose = a, b, c

if __name__ == "__main__":
    MainApp().run()
