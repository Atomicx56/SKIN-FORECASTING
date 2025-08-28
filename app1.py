from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# ========== Config ==========
IMG_SIZE = (256, 256)
Window.size = (1000, 720)
Window.clearcolor = (1, 1, 1, 1)  # White background

# ========== Utilities ==========
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def array_to_texture(array):
    array = (array * 255).astype(np.uint8)
    array = np.clip(array, 0, 255)
    img = Image.fromarray(array)
    texture = Texture.create(size=img.size)
    texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
    texture.flip_vertical()
    return texture

# ========== Main Layout ==========
class SkinForecastGUI(BoxLayout):
    def __init__(self, **kwargs):
        super(SkinForecastGUI, self).__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        self.t1_path = None
        self.t2_path = None
        self.model = load_model("models/generator_epoch150.keras", compile=False)

        # ==== Title ====
        title_box = BoxLayout(size_hint=(1, 0.12), padding=[0, 10, 0, 10])
        with title_box.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.2, 0.4, 0.7, 1)  # Professional blue background
            self.title_bg = Rectangle(pos=title_box.pos, size=title_box.size)
        def update_title_bg(instance, value):
            self.title_bg.pos = title_box.pos
            self.title_bg.size = title_box.size
        title_box.bind(pos=update_title_bg, size=update_title_bg)
        self.title = Label(
            text="[b]Skin Lesion Forecasting[/b]",
            font_size='30sp',
            size_hint=(1, 1),
            color=(1, 1, 1, 1),
            halign='center',
            valign='middle',
            markup=True
        )
        self.title.bind(size=self.title.setter('text_size'))
        title_box.add_widget(self.title)
        self.add_widget(title_box)

        # ==== File Chooser ====
        file_chooser_box = BoxLayout(size_hint=(1, 0.45))
        with file_chooser_box.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.1, 0.1, 0.1, 1)
            self.file_chooser_bg = Rectangle(pos=file_chooser_box.pos, size=file_chooser_box.size)
        def update_bg(instance, value):
            self.file_chooser_bg.pos = file_chooser_box.pos
            self.file_chooser_bg.size = file_chooser_box.size
        file_chooser_box.bind(pos=update_bg, size=update_bg)
        self.file_chooser = FileChooserIconView(path=os.getcwd())
        file_chooser_box.add_widget(self.file_chooser)
        self.add_widget(file_chooser_box)

        # ==== Button Row ====
        btn_layout = BoxLayout(size_hint=(1, 0.12), spacing=10, padding=5)
        self.button_t1 = Button(text=" Select T1", background_color=(0.3, 0.4, 0.9, 1), font_size='16sp')
        self.button_t1.bind(on_press=self.select_t1)

        self.button_t2 = Button(text=" Select T2", background_color=(0.3, 0.4, 0.9, 1), font_size='16sp')
        self.button_t2.bind(on_press=self.select_t2)

        self.predict_button = Button(text=" Predict T3", background_color=(0.2, 0.8, 0.5, 1), font_size='16sp')
        self.predict_button.bind(on_press=self.predict_t3)

        btn_layout.add_widget(self.button_t1)
        btn_layout.add_widget(self.button_t2)
        btn_layout.add_widget(self.predict_button)
        self.add_widget(btn_layout)

        # ==== Image Display Row ====
        self.img_grid = GridLayout(cols=3, size_hint=(1, 0.35), spacing=10, padding=[10, 5])
        self.img_t1 = KivyImage()
        self.img_t2 = KivyImage()
        self.img_pred = KivyImage()
        self.img_grid.add_widget(self._build_image_box(self.img_t1, "T1 Image"))
        self.img_grid.add_widget(self._build_image_box(self.img_t2, "T2 Image"))
        self.img_grid.add_widget(self._build_image_box(self.img_pred, "Predicted T3"))
        self.add_widget(self.img_grid)

        # ==== Status Label ====
        self.status = Label(text=" Select T1 and T2 lesion images to begin.",
                            font_size='16sp',
                            size_hint=(1, 0.05),
                            color=(0.9, 0.9, 0.9, 1))
        self.add_widget(self.status)

    def _build_image_box(self, image_widget, label_text):
        box = BoxLayout(orientation='vertical', spacing=5)
        box.add_widget(image_widget)
        label = Label(text=label_text, size_hint=(1, 0.1), font_size='15sp', color=(1, 1, 1, 1))
        box.add_widget(label)
        return box

    def select_t1(self, instance):
        if self.file_chooser.selection:
            self.t1_path = self.file_chooser.selection[0]
            self.status.text = f"Selected T1: {os.path.basename(self.t1_path)}"
            self.img_t1.source = self.t1_path
            self.img_t1.reload()

    def select_t2(self, instance):
        if self.file_chooser.selection:
            self.t2_path = self.file_chooser.selection[0]
            self.status.text = f" Selected T2: {os.path.basename(self.t2_path)}"
            self.img_t2.source = self.t2_path
            self.img_t2.reload()

    def predict_t3(self, instance):
        if not self.t1_path or not self.t2_path:
            self.status.text = "⚠️ Please select both T1 and T2 images."
            return

        t1_input = preprocess_image(self.t1_path)
        t2_input = preprocess_image(self.t2_path)
        pred = self.model.predict([t1_input, t2_input])[0]

        texture = array_to_texture(pred)
        self.img_pred.texture = texture
        self.status.text = "Prediction complete. Check the T3 image."

# ========== App Runner ==========
class LesionApp(App):
    def build(self):
        return SkinForecastGUI()

if __name__ == '__main__':
    LesionApp().run()
