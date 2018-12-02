import numpy as np
import openvr


cue_offset = np.zeros(3, dtype=np.float64)
offset_adjustment_mode = 0


def toggle_touchpad_fb_ud():
    global offset_adjustment_mode
    offset_adjustment_mode = 1 - offset_adjustment_mode


def cue_position_fb_ud(rAxis):
    if offset_adjustment_mode == 0:
        cue_offset[2] -= 0.008 * rAxis.y
    elif offset_adjustment_mode == 1:
        cue_offset[1] += 0.008 * rAxis.y


def calc_cue_transformation(pose_0, pose_1, out=None):
    if out is None:
        out = np.zeros((4,4), dtype=np.float64)
    r_0, r_1 = pose_0[:,3], pose_1[:,3]
    r_01 = r_1 - r_0
    y_axis = r_01 / np.linalg.norm(r_01)
    out[3,:3] = r_0 + cue_offset[2] * y_axis
    out[3,3]  = 1
    out[1,:3] = y_axis
    x_axis, z_axis = pose_0[:,0], pose_0[:,2]
    dotx, dotz = y_axis.dot(x_axis), y_axis.dot(z_axis)
    if abs(dotx) >= abs(dotz):
        out[2,:3] = z_axis - dotz * y_axis
        out[2,:3] /= np.linalg.norm(out[2,:3])
        out[0,:3] = np.cross(y_axis, out[2,:3])
    else:
        out[0,:3] = x_axis - dotx * y_axis
        out[0,:3] /= np.linalg.norm(out[0,:3])
        out[2,:3] = np.cross(out[0,:3], y_axis)
    return out


def calc_cue_contact_velocity(r_c, r_0, r_1, v_0, v_1):
    r_01 = r_1 - r_0
    v_01 = v_1 - v_0
    omega = np.array(((r_01[1]*v_01[2] - r_01[2]*v_01[1]) / (r_01[1]**2 + r_01[2]**2),
                      (r_01[2]*v_01[0] - r_01[0]*v_01[2]) / (r_01[2]**2 + r_01[0]**2),
                      (r_01[0]*v_01[1] - r_01[1]*v_01[0]) / (r_01[0]**2 + r_01[1]**2)), dtype=np.float64)
    return v_0 + np.cross(omega, r_c - r_0)


axis_callbacks = {
    openvr.k_EButton_Axis0: cue_position_fb_ud,
    #openvr.k_EButton_Axis1: lock_to_cue
}


button_press_callbacks = {
    openvr.k_EButton_Grip           : toggle_touchpad_fb_ud,
    #openvr.k_EButton_ApplicationMenu: toggle_vr_menu,
    #openvr.k_EButton_Grip: reset,
    #openvr.k_EButton_ApplicationMenu: game.advance_time,
}
