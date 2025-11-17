from backend.image_descriptor import ImageDescriptor

def test_describe_image():
  descriptor = ImageDescriptor()
  result = descriptor.describeImage('./backend/tests/cat.jpg')
  assert 'cat' in result
  
def test_array_to_description():
  descriptor = ImageDescriptor()
  result = descriptor.array_to_description([])
  assert 'This image contains nothing' in result
  result = descriptor.array_to_description(['cat'])
  assert 'This image contains: cat.'
  result = descriptor.array_to_description(['cat', 'dog'])
  assert 'This image contains: cat, dog.'