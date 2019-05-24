from tkinter import Canvas, NW, Tk
from PIL import ImageTk
import numpy as np
import json


class ConfigWindow:
    def __init__(self, master, listen_fun, image_path):
        self.master = master
        self.listen_fun = listen_fun
        master.title("ConfigMaker")

        self.img = ImageTk.PhotoImage(file=image_path)

        self.image = Canvas(master, width=self.img.width()+20, height=self.img.height()+20)
        self.image.create_image(10, 10, anchor=NW, image=self.img)

        self.image.bind("<Button-1>", func=self.__click)
        self.image.pack()

        self.click_count = 0
        self.config = {}
        self.temp_config = {}

    def __click(self, event):
        print(event.x, event.y, 'click')
        if self.click_count == 0:
            self.temp_config['center'] = [event.x, event.y]

            self.config['Cx'] = event.x
            self.config['Cy'] = event.y
            self.click_count += 1
        elif self.click_count == 1:
            self.config['Ri'] = self.__get_dist(self.temp_config['center'], [event.x, event.y])
            self.click_count += 1
        else:
            self.config['Ro'] = self.__get_dist(self.temp_config['center'], [event.x, event.y])
            print(self.config)
            self.master.quit()
            self.listen_fun(self.config)

        self.image.create_rectangle(event.x-1, event.y-1, event.x+1, event.y+1, fill='red', outline="red")

    def __get_dist(self, x1, x2):
        vec1, vec2 = np.array(x1), np.array(x2)
        dif = vec1-vec2
        return int(np.round(np.sqrt(dif.dot(dif))))


class ConfigJsonMaker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.root = Tk()
        self.win = ConfigWindow(self.root, self._write_config, image_path)
        self.root.mainloop()

    def _write_config(self, config):
        with open("config.json", "w") as write_file:
            json.dump(config, write_file)

if __name__ == '__main__':
    win = ConfigJsonMaker("C:\\DataScience\\TestGui\\1.png")