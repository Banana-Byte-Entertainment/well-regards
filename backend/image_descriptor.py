class ImageDescriptor:
  def describeImage(self, img):
    return []
  
  # takes in array of string and returns a string
  # "This image contains: object1, ..., objectN."
  def array_to_description(self, arr):
    result = "This image contains: "
    
    if len(arr) == 0:
      return "This image contains nothing." 

    result += ", ".join(arr)
    result += "."

    return result