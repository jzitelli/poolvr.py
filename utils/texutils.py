import numpy as np
import PIL.Image as Image

width, height = 512, 512
multisample = 4
_width, _height = multisample * width, multisample * height
img = np.zeros((_width, _height, 3), dtype=np.uint8)
# Draw something (http://stackoverflow.com/a/10032271/562769)
x = np.linspace(-1, 1, _width)
y = np.linspace(-1, 1, _height)
xv, yv = np.meshgrid(x, y)
#xx, yy = np.mgrid[:_width, :_height]
#circle = (xx - _width//2 + 0.5)**2 + (yy - height//2 + 0.5)**2
# Set the RGB values
img[xv**2 + yv**2 < 0.975**2*np.ones(xv.shape),:3] = (255, 255, 255)
# for y in range(img.shape[0]):
#     for x in range(img.shape[1]):
#         #r, g, b = circle[y][x], circle[y][x], circle[y][x]
#         #if circle[y][x] < (_width//2-multisample)**2:
#         if xy[y,x]**2) < 0.98**2:
#             r, g, b = 255, 255, 255
#         else:
#             r, g, b = 0, 0, 0
#         img[y][x][0] = r
#         img[y][x][1] = g
#         img[y][x][2] = b
####################################
image = Image.new('RGB', (_width, _height))
image.putdata([(int(r), int(g), int(b)) for (r,g,b) in img.reshape(-1,3)])
image = image.resize((width, height), resample=Image.LANCZOS)
image.save('out.png')
