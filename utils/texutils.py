import logging
import numpy as np
import PIL.Image as Image

_logger = logging.getLogger(__name__)

def gen_sphere_billboards(width=256, height=256, multisample=4,
                          fp_alpha=None, fp_normal=None, fp_depth=None, fp_uv=None):
    _width, _height = multisample * width, multisample * height
    x, y = np.linspace(-1, 1, _width), np.linspace(-1, 1, _height)
    xv, yv = np.meshgrid(x, y)
    mask = xv**2 + yv**2 < 0.975**2*np.ones(xv.shape)

    if fp_alpha:
        img = np.zeros((_width, _height, 4), dtype=np.uint8)
        img[mask,:4] = (255, 255, 255, 255)
        image = Image.new('RGBA', (_width, _height))
        # is there a faster way to do this?
        image.putdata([(int(r), int(g), int(b), int(a)) for (r,g,b,a) in img.reshape(-1,4)])
        image = image.resize((width, height), resample=Image.LANCZOS)
        image.save(fp_alpha)
        _logger.info('wrote alpha texture to "%s"', fp_alpha)

    if fp_normal:
        normals = np.zeros((_width, _height, 3))
        normals[...,0] = xv
        normals[...,1] = yv
        # normals[...,2] = np.sqrt(0.975**2 - xv**2 - yv**2)
        normals[mask,2] = np.sqrt(0.975**2*np.ones(xv.shape)[mask] - xv[mask]**2 - yv[mask]**2)
        # normals = np.nan_to_num(normals)
        normals[~mask,0] = xv[~mask] / np.sqrt(xv[~mask]**2 + yv[~mask]**2)
        normals[~mask,1] = yv[~mask] / np.sqrt(xv[~mask]**2 + yv[~mask]**2)
        normals[~mask,2] = 0
        # normals[xv**2 + yv**2 > 0.975**2*np.ones(xv.shape),1] = 1
        # normals[xv**2 + yv**2 > 0.975**2*np.ones(xv.shape),:3] = -1
        image = Image.new('RGB', (_width, _height))
        image.putdata([(int((x+1)/2*255), int((-y+1)/2*255), int((z+1)/2*255)) for (x,y,z) in normals.reshape(-1,3)])
        image = image.resize((width, height), resample=Image.LANCZOS)
        image.save(fp_normal)
        _logger.info('wrote normal texture to "%s"', fp_normal)
