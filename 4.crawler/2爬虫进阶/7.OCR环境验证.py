
import tesserocr
from PIL import Image

image=Image.open("data/ocr_test.png")
print(tesserocr.image_to_text(image))
