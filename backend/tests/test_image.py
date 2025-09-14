import os
from PIL import Image

class TestImage:
    image_folder = "media"
    image_filename = "test_image.jpg"
    image_path = os.path.join(image_folder, image_filename)
    img = Image.open(image_path)
    
    def return_image_width(self, image):
        return image.size[0]
    
    def return_image_height(self, image):
        return image.size[1]

    def test_answer(self):
        assert self.return_image_width(self.img) == 3000
        assert self.return_image_height(self.img) == 4000