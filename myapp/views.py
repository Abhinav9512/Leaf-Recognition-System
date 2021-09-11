import numpy as np
import cv2
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.preprocessing import image
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage


CATEGORIES = ["acer_palmatum", "betula_lenta" ,"cedrus_libani","diospyros_virginiana","evodia_danielli","ginkgo_biloba","ilex_opaca","juglans_nigra","koelreuteria_paniculata","malus_pumila","ostrya_virginiana","pinus_taeda","quercus_palustris","ulmus_pumila", "zelkova_serrata"]  # will use this to convert prediction num to string value

model = load_model('model.h5')

model.make_predict_function()

# Create your views here.
def predict_label(img_path):
  IMG_SIZE = 256  # 50 in txt-based
  img_array = cv2.imread(img_path)  # read in the image, convert to grayscale
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
  # new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  new_array= np.reshape(new_array, (-1, IMG_SIZE, IMG_SIZE, 3))
  p = model.predict(new_array)
  return CATEGORIES[int(np.argmax(p[0]))]
    # i = image.load_img(img_path, target_size=(256,256))
    # i = i.reshape(1, 256,256,3)
    # p = model.predict_classes(i)
    # return CATEGORIES[p[0]]


@csrf_exempt
def index(request):
    if request.method == "POST":
        img = request.FILES['my_image']
        img_path = "static/" + str(img.name)
        # img.save(img_path)
        fs = FileSystemStorage()
        filename = fs.save(img_path, img)
        uploaded_file_url = fs.url(filename)
        print(filename)
        # print(uploaded_file_url)
        p = predict_label("media/"+filename)

        # return render(request, "index.html", prediction = p, img_path = filename)

        context = {'prediction' : p, 'img_path' : "media/"+filename}
        return render(request, 'index.html', context) 
    else:
        return render(request,"index.html")

 
@csrf_exempt
def main(request):
    return render(request,"index.html")

