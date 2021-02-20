from tkinter import *
import subprocess
from PIL import Image
import threading
import tkinter as tk
from tkinter.ttk import Combobox

window = Tk()
window.geometry('1000x500')
window.title('Визуализация работы персептрона')
window.configure(bg='grey19')
window.resizable(width=True, height=False)
hand_writing = Canvas(window, bg='white', height=300, width=300)
hand_writing.place(x=50, y=50)
brush_size = 10
brush_color = "black"


def draw(event):
    hand_writing.create_oval(event.x - brush_size,
                             event.y - brush_size,
                             event.x + brush_size,
                             event.y + brush_size,
                             fill=brush_color, outline=brush_color)


def clear_paintings():
    hand_writing.delete("all")


was_already_imported = False


def run_action():
    global was_already_imported
    import importlib
    if not was_already_imported:
        import Network
        was_already_imported = True
    else:
        Network = importlib.import_module("Network")
    print(Network.ANSWER)


def start_learning():
    learning_lbl = Label(window, text="Идёт обучение...", font=('Monsterrat', 20), bg='grey19', fg='White')
    learning_lbl.place(x=750, y=20)
    thread = threading.Thread(target=run_action)
    learn_btn.config(state=tk.DISABLED)
    learn_btn.configure(bg='grey30')
    thread.start()

    def check_thread(thr):
        if thr.is_alive():
            learn_btn.after(100, lambda: check_thread(thr))
        else:
            learn_btn.config(state=tk.NORMAL)
            learn_btn.configure(bg='white', fg='grey19')
            learning_lbl.place(x=700, y=20)
            learning_lbl.configure(text='Обучение завершено!')

    check_thread(thread)


def save_canvas():
    hand_writing.postscript(
        file=r"D:\Python\Предпроф\ImageMagick-7.0.11-0-portable-Q16-HDRI-x64\to_recognise.ps",
        colormode="color")

    cmd = r'D:\Python\Предпроф\ImageMagick-7.0.11-0-portable-Q16-HDRI-x64\magick.exe ' \
          r'D:\Python\Предпроф\ImageMagick-7.0.11-0-portable-Q16-HDRI-x64\to_recognise.ps ' \
          r'D:\Python\Предпроф\ImageMagick-7.0.11-0-portable-Q16-HDRI-x64\to_recognise.jpg'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0]
    print(result.decode('cp866'))
    img = Image.open(
        r'D:\Python\Предпроф\ImageMagick-7.0.11-0-portable-Q16-HDRI-x64\to_recognise.jpg')
    width = 28
    height = 28
    resized_img = img.resize((width, height), Image.ANTIALIAS)
    resized_img.save('resized.jpg')


hand_writing.bind("<B1-Motion>", draw)

cls_paintings = Button(window, text=' Очистить холст ', command=clear_paintings)
cls_paintings.configure(font=('Monsterrat', 14), bg='grey45', fg='white')
cls_paintings.place(x=120, y=360)

save_paintings = Button(window, text='Сохранить холст', command=save_canvas)
save_paintings.configure(font=('Monsterrat', 14), bg='grey45', fg='white')
save_paintings.place(x=120, y=420)

learn_btn = Button(window, text='Запустить обучение', command=start_learning)
learn_btn.configure(font=('Monsterrat', 14), bg='white', fg='grey19')
learn_btn.place(x=800, y=420)

num_of_hidden_layers = Spinbox(window, from_=1, to=6)
num_of_hidden_layers.place(x=800, y=180)
#
settings = Label(window, text="Настройки нейросети", font=('Monsterrat', 20), bg='grey19', fg='White')
settings.place(x=550, y=50)

activation_func = Label(window, text="Функции активации слоев", font=('Monsterrat', 15), bg='grey19', fg='White')
activation_func.place(x=400, y=110)

layers_choose = Label(window, text="Кол-во скрытых слоёв", font=('Monsterrat', 15), bg='grey19', fg='White')
layers_choose.place(x=780, y=110)

num_epochs = Label(window, text="Кол-во эпох обучения", font=('Monsterrat', 15), bg='grey19', fg='White')
num_epochs.place(x=400, y=260)

num_of_hidden_layers = Spinbox(window, from_=1, to=8)
num_of_hidden_layers.place(x=420, y=300)


IN = Label(window, text="Входного", font=('Monsterrat', 10), bg='grey19', fg='White')
IN.place(x=400, y=150)
HID = Label(window, text="Скрытого", font=('Monsterrat', 10), bg='grey19', fg='White')
HID.place(x=525, y=150)
OUT = Label(window, text="Выходного", font=('Monsterrat', 10), bg='grey19', fg='White')
OUT.place(x=655, y=150)

funcIN = Combobox(window, width=15)
funcIN.place(x=370, y=180)

funcHID = Combobox(window, width=15)
funcHID.place(x=500, y=180)

funcOUT = Combobox(window, width=15)
funcOUT.place(x=630, y=180)

funcIN['values'] = ("Deserialize()", "Elu()", "Exponential()",
                    "Gelu()", "Get()", "Hard_sigmoid()", "Linear()", "Relu()", "Selu()",
                    "Serialize()", "Sigmoid()", "Softmax()", "Softplus()", "Softsign()"
                    , "Swish()", "Tanh()")

funcHID['values'] = ("Deserialize()", "Elu()", "Exponential()",
                     "Gelu()", "Get()", "Hard_sigmoid()", "Linear()", "Relu()", "Selu()",
                     "Serialize()", "Sigmoid()", "Softmax()", "Softplus()", "Softsign()"
                     , "Swish()", "Tanh()")

funcOUT['values'] = ("Deserialize()", "Elu()", "Exponential()",
                     "Gelu()", "Get()", "Hard_sigmoid()", "Linear()", "Relu()", "Selu()",
                     "Serialize()", "Sigmoid()", "Softmax()", "Softplus()", "Softsign()"
                     , "Swish()", "Tanh()")

window.mainloop()
