from backend.read_image import ObjectDetector

class ImageDescriptor:

  # takes in PATH to image, returns list of detected objects
  def describeImage(self, img_path):
    img_reader = ObjectDetector(confidence_threshold=0.7)
    object_names = []
    for detection in img_reader.detect_objects(img_path):
      object_names.append(detection['class'])
    return object_names
  
  # takes in array of string and returns a string
  # "This image contains: object1, ..., objectN."
  def array_to_description(self, arr):
    result = "This image contains: "
    
    if len(arr) == 0:
      return "This image contains nothing." 

    result += ", ".join(arr)
    result += "."

    return result