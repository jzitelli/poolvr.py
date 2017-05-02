import logging
import bpy


_logger = logging.getLogger(__name__)


def open_blender_file(filepath):
    ret = bpy.ops.wm.open_mainfile(filepath=filepath)
    _logger.info(ret)

def toggle_edit_mode():
    bpy.ops.object.editmode_toggle()

def toggle_select_faces():
    bpy.ops.mesh.select_mode(type='FACE')

def select_all_faces():
    bpy.ops.mesh.select_all()

def add_hexa_primitive(hexa, swap_yz=True, scale=1.0):
    # [(0, Vector((1.0, 0.9999999403953552, -1.0))),
    #  (1, Vector((1.0, -1.0, -1.0))),
    #  (2, Vector((-1.0000001192092896, -0.9999998211860657, -1.0))),
    #  (3, Vector((-0.9999996423721313, 1.0000003576278687, -1.0))),
    #  (4, Vector((1.0000004768371582, 0.999999463558197, 1.0))),
    #  (5, Vector((0.9999993443489075, -1.0000005960464478, 1.0))),
    #  (6, Vector((-1.0000003576278687, -0.9999996423721313, 1.0))),
    #  (7, Vector((-0.9999999403953552, 1.0, 1.0)))]
    bpy.ops.mesh.primitive_cube_add()
    mesh = bpy.data.meshes.values()[-1]
    for v in mesh.vertices:
        v.co *= scale

if __name__ == "__main__":
    add_hexa_primitive(None, swap_yz=True, scale=1.2)
    add_hexa_primitive(None, swap_yz=True, scale=1.4)
    add_hexa_primitive(None, swap_yz=True, scale=1.8)
    #bpy.ops.mesh.select_mode(type='VERT')
    #bpy.ops.mesh.select_all()
    toggle_edit_mode()
    toggle_select_faces()
    select_all_faces()
    #bpy.ops.mesh.select_all()
    #bpy.ops.uv.unwrap()
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES', PREF_NEW_UVLAYER=True, PREF_APPLY_IMAGE=True)
    bpy.ops.object.bake_image()
    bpy.data.images.values()[-1].save_render('pybake2.png')
    #bpy.ops.image.save_as(save_as_render=False, filepath='pybake.png', relative_path=True)
    #bpy.ops.object.mode_set()
    #bpy.ops.material.new()
    #bpy.ops.texture.new()
    #bpy.ops.image.save_dirty()
    #bpy.context.object.active_material.texture_slots[0].uv_layer = "UVMap"
    #bpy.context.object.active_material.use_shadeless = True
    # image = bpy.data.images.values()[-1]
    # image.filepath = 'bake.png'
    # image.save()
