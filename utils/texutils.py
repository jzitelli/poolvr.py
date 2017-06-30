import numpy as np
import PIL.Image as Image

width, height = 128, 128
multisample = 4
_width, _height = multisample * width, multisample * height


img = np.zeros((_width, _height, 4), dtype=np.uint8)
x = np.linspace(-1, 1, _width)
y = np.linspace(-1, 1, _height)
xv, yv = np.meshgrid(x, y)
img[xv**2 + yv**2 < 0.975**2*np.ones(xv.shape),:4] = (255, 255, 255, 255)
image = Image.new('RGBA', (_width, _height))
image.putdata([(int(r), int(g), int(b), int(a)) for (r,g,b,a) in img.reshape(-1,4)])
image = image.resize((width, height), resample=Image.LANCZOS)
image.save('mask.png')


normals = np.zeros((_width, _height, 3))

normals[...,0] = xv
normals[...,1] = yv
normals[...,2] = np.sqrt(0.975**2 - xv**2 - yv**2)
normals = np.nan_to_num(normals)
normals[xv**2 + yv**2 > 0.975**2*np.ones(xv.shape),:3] = -1

image = Image.new('RGB', (_width, _height))
image.putdata([(int((x+1)/2*255), int((y+1)/2*255), int((z+1)/2*255)) for (x,y,z) in normals.reshape(-1,3)])
image.resize((width, height), resample=Image.LANCZOS)
image.save('normals.png')
