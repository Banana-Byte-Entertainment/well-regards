import os
from PIL import Image

class TestImage:
    image_folder = "backend/tests/media"
    image_filename = "test_image.jpg"
    image_path = os.path.join(image_folder, image_filename)
    img = Image.open(image_path)
    
    def return_image_width(self, image):
        return image.size[0]
    
    def return_image_height(self, image):
        return image.size[1]
    
    def return_image_size(self, image):
        return image.size
    
    def return_image_filesize(self, image):
        return os.stat(self.image_path).st_size

    def test_size(self):
        assert self.return_image_width(self.img) == 3000
        assert self.return_image_height(self.img) == 4000
        assert self.return_image_size(self.img) == (3000, 4000)
        
    def test_filesize(self):
        assert self.return_image_filesize(self.img) == 490741
        
        