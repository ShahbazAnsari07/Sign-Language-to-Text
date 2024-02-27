from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys, os
import matplotlib.pyplot as plt
import hunspell
from string import ascii_uppercase

class Application:
    def __init__(self):
        self.directory = 'model'
        self.hs = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load models
        self.loaded_model = self.load_model("model-bw")
        self.loaded_model_dru = self.load_model("model-bw_dru")
        self.loaded_model_tkdi = self.load_model("model-bw_tkdi")
        self.loaded_model_smn = self.load_model("model-bw_smn")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")

        # Initialize image panels
        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)

        # Text labels
        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 40, "bold"))
        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Character:", font=("Courier", 40, "bold"))
        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=220, y=700)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=700)
        self.T2.config(text="Word:", font=("Courier", 40, "bold"))
        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=350, y=760)
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=760)
        self.T3.config(text="Sentence:", font=("Courier", 40, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=820)
        self.T4.config(text="Suggestions", fg="red", font=("Courier", 40, "bold"))

        # Buttons
        self.btcall = tk.Button(self.root, command=self.action_call, height=0, width=0)
        self.btcall.config(text="About", font=("Courier", 14))
        self.btcall.place(x=825, y=0)

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=890)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=890)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=890)

        self.bt4 = tk.Button(self.root, command=self.action4, height=0, width=0)
        self.bt4.place(x=125, y=950)

        self.bt5 = tk.Button(self.root, command=self.action5, height=0, width=0)
        self.bt5.place(x=425, y=950)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def load_model(self, model_name):
        json_file = open(f"{self.directory}/{model_name}.json", "r")
        model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(f"{self.directory}/{model_name}.h5")
        return loaded_model

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))
            predicts = self.hs.suggest(self.word)
            self.update_suggestions(predicts)
        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {'blank': result[0][0]}
        for i, letter in enumerate(ascii_uppercase):
            prediction[letter] = result[0][i + 1]

        # LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction1[0][0] if prediction1[0][0] == 'S' else prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def update_suggestions(self, predicts):
        for idx, button in enumerate([self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]):
            if idx < len(predicts):
                button.config(text=predicts[idx], font=("Courier", 20))
            else:
                button.config(text="")

    def action1(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]

    def action2(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 3:
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 4:
            self.word = ""
            self.str += " "
            self.str += predicts[4]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self):
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")

        # Developers information
        developers_info = [
            {"name": "RC", "roll": "IIT2016141", "image": "ravi.png"},
            {"name": "Nitin", "roll": "IIT2016132", "image": "nitin.png"},
            {"name": "Luv", "roll": "IIT2016085", "image": "luv.png"},
            {"name": "Sheldon", "roll": "IIT2016137", "image": "sheldon.png"},
            {"name": "Siddhant", "roll": "IIT2016069", "image": "sid.png"}
        ]

        # Display developers
        for idx, info in enumerate(developers_info):
            photo = tk.PhotoImage(file=f'Pictures/{info["image"]}')
            label = tk.Label(self.root1, image=photo)
            label.image = photo
            label.place(x=20 + idx * 180, y=105)
            tk.Label(self.root1, text=f"{info['name']}\n{info['roll']}", font=("Courier", 15, "bold")).place(x=20 + idx * 180, y=250)

        # Supervisor information
        tk.Label(self.root1, text="Under the supervision of", fg="red", font=("Courier", 30, "bold")).place(x=170, y=360)
        photo = tk.PhotoImage(file='Pictures/sir.png')
        label = tk.Label(self.root1, image=photo)
        label.image = photo
        label.place(x=350, y=420)
        tk.Label(self.root1, text="Dr. Vrijendra Singh", font=("Courier", 30, "bold")).place(x=230, y=670)

        self.root1.mainloop()


print("Starting Application...")
pba = Application()
pba.root.mainloop()
