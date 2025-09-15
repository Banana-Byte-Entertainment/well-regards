from backend.image_descriptor import ImageDescriptor

def test_describe_image():
  descriptor = ImageDescriptor()
  result = descriptor.describeImage('./backend/tests/cat.jpg')
  assert 'cat' in result