import logging

_logger = logging.getLogger(__name__)


try:
    import bpy
except ImportError as err:
    _logger.error('could not import Blender Python module ("bpy"): %s', err)
    bpy = None


def load_scene(fp_scene):
    bpy.ops.wm.open_mainfile(filepath=fp_scene)
    _logger.info('opened scene file "%s"', fp_scene)


def bake_all_faces(fp_image, fp_scene=None, mode="full render"):
    if bpy is None:
        return
    bpy.ops.object.editmode_toggle()
    #bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all()
    #bpy.ops.uv.unwrap()
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES', PREF_NEW_UVLAYER=True, PREF_APPLY_IMAGE=True)
    _logger.info('baking %s...', mode)
    bpy.ops.object.bake_image()
    _logger.info('...done')
    bpy.data.images.values()[-1].save_render(fp_image)
    _logger.info('wrote image to "%s"', fp_image)


    #bpy.ops.image.save_as(save_as_render=False, filepath='pybake.png', relative_path=True)
    #bpy.ops.object.mode_set()

def create_material():
    if bpy is None:
        return
    bpy.ops.material.new()
    bpy.ops.texture.new()
    bpy.ops.image.save_dirty()
    # image = bpy.data.images.values()[-1]
    # image.filepath = 'bake.png'
    # image.save()
    #bpy.context.object.active_material.texture_slots[0].uv_layer = "UVMap"
    #bpy.context.object.active_material.use_shadeless = True
