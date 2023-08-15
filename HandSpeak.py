import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter
import pyttsx3
from keras.models import Sequential
from keras.layers import LSTM, Dense

class HandSpeakWindow:
    def __init__(self, appearance_mode='light', title='Hand-Speak', geometry='400x600'):

        ctk.set_appearance_mode(appearance_mode)

        self.window = tkinter.Tk()
        self.window.title(title)
        self.window.geometry(geometry)

        self.label = ctk.CTkLabel(
            self.window,
            text='Hand Speak',
            font=('Roboto', 24),
            pady=30,
            padx=30,
            corner_radius=10
        )
        self.label.pack()

        self.button = ctk.CTkButton(
            self.window,
            text='Start',
            hover_color='#AA0',
            command=lambda: self.run_detection()
        )
        self.button.pack()

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_action = None
        self.last_action_update_time = None
        self.actions = np.array(['Yes', 'What', 'Thank you', 'Take-out', 'Please', 'No', 'How Much', 'Hello', 'Food', 'Dine-in'])
        self.no_sequences = 30
        self.sequence_length = 30
        self.sequence = []
        self.engine = pyttsx3.init()
        self.threshold = 0.56
        self.predicted_word = None

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.load_weights('FINAL_MODEL_C.h5')
        self.is_capturing = False
        self.capturing_start_time = None
    def mediapipe_detection(self, image, model):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                       self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        spoken = False
        previous_word = None
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                # Check if capturing is ongoing
                if self.is_capturing:
                    if self.capturing_start_time is None:
                        self.capturing_start_time = time.time()

                    # Calculate remaining time for capturing
                    remaining_time = 3 - (time.time() - self.capturing_start_time)
                    if remaining_time <= 0:
                        self.is_capturing = False
                        self.capturing_start_time = None
                        res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        self.predicted_word = self.actions[np.argmax(res)]
                        self.engine.say(self.predicted_word)
                        self.engine.runAndWait()

                    # Render capturing message with remaining time
                    message = f"Capturing for {max(int(remaining_time), 0)}"
                    cv2.putText(image, message, (int(image.shape[1]/2) - 25, int(image.shape[0]/2) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                if self.predicted_word is not None:
                    cv2.putText(image, self.predicted_word, (int(image.shape[1]/2 - 20*len(self.predicted_word)/2), 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if self.predicted_word != previous_word:
                    if self.last_action != self.predicted_word or (self.last_action_update_time is not None and time.time() - self.last_action_update_time > 1.5):
                        self.engine.say(self.predicted_word)
                        self.engine.runAndWait()
                        self.last_action = self.predicted_word
                        self.last_action_update_time = time.time()
                        previous_word = self.predicted_word

                cv2.imshow('OpenCV Feed', image)

                # Listen for key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.is_capturing = True
                    self.capturing_start_time = None
                    self.predicted_word = None
                    self.last_action = None
                    self.last_action_update_time = None

        cap.release()
        cv2.destroyAllWindows()


    def run(self):
        self.window.mainloop()

# Create an instance of the HandSpeakWindow class
hand_speak_window = HandSpeakWindow()

# Run the window
hand_speak_window.run()
