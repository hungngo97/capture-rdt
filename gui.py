# import the necessary packages
from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import subprocess


def resize_image(img, gray=True, scale_percent=400):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def select_image():

    # grab a reference to the image panels
    global panelA, panelB, croppedPanel

    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()
    # ensure a file path was selected
    done = False

    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, False, 40)
        edged = resize_image(edged, True, 40)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
        # if the panels are None, initialize them
        subprocess.call("python main.py --f " + str(path), shell=True)
        # Read image from directory
        edged = cv2.imread('result.png', cv2.IMREAD_UNCHANGED)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        edged = resize_image(edged, False, 70)
        edged = Image.fromarray(edged)
        edged = ImageTk.PhotoImage(edged)

        cropped = cv2.imread('cropResult.png', cv2.IMREAD_UNCHANGED)
        print('[INFO] cropped shape', cropped.shape)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = Image.fromarray(cropped)
        cropped = ImageTk.PhotoImage(cropped)

        # And then display
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="top", padx=10, pady=10)

            textLabel1 = Label(text="Original Image" + str(path))
            textLabel1.pack(side="top")

            # the first panel will store our original image
            croppedPanel = Label(image=cropped)
            croppedPanel.image = cropped
            croppedPanel.pack(side="top", padx=10, pady=10)

            textLabel3 = Label(text="Cropped Image")
            textLabel3.pack(side="top")

            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="bottom", padx=10, pady=10)

            textLabel2 = Label(text="Result Image")
            textLabel2.pack(side="top")

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            croppedPanel.configure(image=cropped)
            panelA.image = image
            panelB.image = edged
            croppedPanel.image = cropped


# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
croppedPanel = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
