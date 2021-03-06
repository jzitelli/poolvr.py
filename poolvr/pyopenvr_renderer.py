from ctypes import c_float, cast, POINTER
from contextlib import contextmanager
import logging
_logger = logging.getLogger(__name__)
import numpy as np
from numpy import dot, array, asarray
from numpy.ctypeslib import as_array
import OpenGL.GL as gl
import openvr
from openvr.gl_renderer import OpenVrFramebuffer as OpenVRFramebuffer
from openvr.gl_renderer import matrixForOpenVrMatrix as matrixForOpenVRMatrix


c_float_p = POINTER(c_float)


class OpenVRRenderer(object):
    def __init__(self, multisample=0, znear=0.1, zfar=1000, window_size=(960,1080)):
        self.multisample = multisample
        self.znear, self.zfar = znear, zfar
        self.window_size = np.array(window_size, dtype=np.int64)
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self.poses = poses_t()
        self.eye_matrices = (np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        self.camera_matrices = (np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        self.hmd_matrix = np.eye(4, dtype=np.float32)
        self.hmd_matrix_inv = np.eye(4, dtype=np.float32)
        self.vr_event = openvr.VREvent_t()
        self._controller_indices = []
        self._controller_poll_interval = 0.25
        self._nframes = 0
        self._time_to_poll = 0.0

    def init_gl(self, clear_color=(0.0, 0.0, 0.0, 0.0)):
        self.vr_system = openvr.init(openvr.VRApplication_Scene)
        w, h = self.vr_system.getRecommendedRenderTargetSize()
        self.render_target_size = np.array((w, h), dtype=np.float32)
        self.vr_framebuffers = (OpenVRFramebuffer(w, h, multisample=self.multisample),
                                OpenVRFramebuffer(w, h, multisample=self.multisample))
        self.vr_compositor = openvr.VRCompositor()
        if self.vr_compositor is None:
            raise Exception('unable to create compositor')
        self.vr_framebuffers[0].init_gl()
        self.vr_framebuffers[1].init_gl()
        self.update_projection_matrix()
        self.eye_to_head_transforms = (asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Left))),
                                       asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Right))))
        self.eye_transforms = (asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Left)).I),
                               asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Right)).I))
        gl.glClearColor(*clear_color)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def update_projection_matrix(self):
        znear, zfar = self.znear, self.zfar
        self.projection_matrices = (asarray(matrixForOpenVRMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Left, znear, zfar))),
                                    asarray(matrixForOpenVRMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Right, znear, zfar))))
        self.projection_lrbts = (array(self.vr_system.getProjectionRaw(openvr.Eye_Left)),
                                 array(self.vr_system.getProjectionRaw(openvr.Eye_Right)))


    @contextmanager
    def render(self, meshes=None, **frame_data):
        self.vr_compositor.waitGetPoses(self.poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
        _hmd_pose = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not _hmd_pose.bPoseIsValid:
            yield None
            return
        hmd_pose = as_array(cast(_hmd_pose.mDeviceToAbsoluteTracking.m, c_float_p),
                            shape=(3,4))
        hmd_velocity = as_array(_hmd_pose.vVelocity.v)
        hmd_angular_velocity = as_array(_hmd_pose.vAngularVelocity.v)
        poses, velocities, angular_velocities = [], [], []
        for i in self._controller_indices:
            controller_pose = self.poses[i]
            if controller_pose.bPoseIsValid:
                poses.append(as_array(cast(controller_pose.mDeviceToAbsoluteTracking.m, c_float_p),
                                      shape=(3,4)))
                velocities.append(as_array(controller_pose.vVelocity.v))
                angular_velocities.append(as_array(controller_pose.vAngularVelocity.v))
        hmd_matrix = self.hmd_matrix
        hmd_matrix_inv = self.hmd_matrix_inv
        hmd_matrix[:,:3] = hmd_pose.T
        hmd_matrix_inv[:3,:3] = hmd_pose[:,:3]
        dot(hmd_matrix[:3,:3], hmd_matrix[3,:3], out=hmd_matrix_inv[3,:3])
        hmd_matrix_inv[3,:3] *= -1
        for eye in (0,1):
            dot(hmd_matrix_inv, self.eye_transforms[eye], out=self.eye_matrices[eye])
            dot(self.eye_to_head_transforms[eye], hmd_matrix, out=self.camera_matrices[eye])
        frame_data.update({
            'hmd_pose': hmd_pose,
            'hmd_velocity': hmd_velocity,
            'hmd_angular_velocity': hmd_angular_velocity,
            'controller_indices': self._controller_indices,
            'controller_poses': poses,
            'controller_velocities': velocities,
            'controller_angular_velocities': angular_velocities,
            'camera_matrices': self.camera_matrices,
            'eye_matrices': self.eye_matrices,
            'projection_matrices': self.projection_matrices,
            'projection_lrbts': self.projection_lrbts,
            'window_size': self.render_target_size,
            'iResolution': self.render_target_size,
            'znear': self.znear,
            'zfar': self.zfar
        })
        # if self._nframes % 90 == 0: _logger.debug('yielding frame_data:\n%s\n', '\n'.join('%s:\n%s' % it for it in frame_data.items()))
        yield frame_data

        for eye in (0,1):
            gl.glViewport(0, 0, self.vr_framebuffers[eye].width, self.vr_framebuffers[eye].height)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.vr_framebuffers[eye].fb)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            frame_data['view_matrix'] = self.eye_matrices[eye]
            frame_data['camera_matrix'] = self.camera_matrices[eye]
            frame_data['projection_matrix'] = self.projection_matrices[eye]
            frame_data['projection_lrbt'] = self.projection_lrbts[eye]
            # if self._nframes % 90 == 0: _logger.debug('drawing for eye %d with frame_data:\n%s\n', eye, '\n'.join('%s:\n%s' % item for item in frame_data.items()))
            if meshes is not None:
                for mesh in meshes:
                    mesh.draw(**frame_data)
        #self.vr_compositor.submit(openvr.Eye_Left, self.vr_framebuffers[0].texture)
        #self.vr_compositor.submit(openvr.Eye_Right, self.vr_framebuffers[1].texture)
        self.vr_framebuffers[0].submit(openvr.Eye_Left)
        self.vr_framebuffers[1].submit(openvr.Eye_Right)
        # mirror left eye framebuffer to screen:
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,
                             self.vr_framebuffers[0].resolve_fb if self.vr_framebuffers[0].multisample
                             else self.vr_framebuffers[0].fb)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glBlitFramebuffer(0, 0, self.vr_framebuffers[0].width, self.vr_framebuffers[0].height,
                             0, 0, self.window_size[0], self.window_size[1],
                             gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT,
                             gl.GL_NEAREST)
        self._nframes += 1

    def process_input(self, dt, button_press_callbacks=None, axis_callbacks=None):
        if len(self._controller_indices) < 2:
            if self._time_to_poll <= 0.0:
                self._poll_for_controllers()
                self._time_to_poll = self._controller_poll_interval
            else:
                self._time_to_poll -= dt
        if self._controller_indices:
            for ii, i in enumerate(self._controller_indices):
                got_state, state = self.vr_system.getControllerState(i)
                if got_state and axis_callbacks and state.ulButtonTouched:
                    if state.ulButtonTouched == 4294967296 and openvr.k_EButton_Axis0 in axis_callbacks:
                        axis_callbacks[openvr.k_EButton_Axis0](state.rAxis[0])
            if self.vr_system.pollNextEvent(self.vr_event) \
               and button_press_callbacks \
               and self.vr_event.eventType == openvr.VREvent_ButtonPress:
                    button = self.vr_event.data.controller.button
                    if button in button_press_callbacks:
                        button_press_callbacks[button]()
                # elif self.vr_event.eventType == openvr.VREvent_ButtonUnpress:
                #     pass

    def shutdown(self):
        openvr.shutdown()

    def _poll_for_controllers(self):
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
                self._controller_indices.append(i)
