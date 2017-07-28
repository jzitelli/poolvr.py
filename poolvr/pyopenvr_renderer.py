from ctypes import c_float, cast, POINTER
from contextlib import contextmanager
import numpy as np
import OpenGL.GL as gl
import openvr
from openvr.gl_renderer import OpenVrFramebuffer as OpenVRFramebuffer
from openvr.gl_renderer import matrixForOpenVrMatrix as matrixForOpenVRMatrix


c_float_p = POINTER(c_float)


class OpenVRRenderer(object):
    def __init__(self, multisample=0, znear=0.1, zfar=1000, window_size=(960,1080)):
        self.vr_system = openvr.init(openvr.VRApplication_Scene)
        w, h = self.vr_system.getRecommendedRenderTargetSize()
        self.window_size = window_size
        self.multisample = multisample
        self.vr_framebuffers = (OpenVRFramebuffer(w, h, multisample=multisample),
                                OpenVRFramebuffer(w, h, multisample=multisample))
        self.vr_compositor = openvr.VRCompositor()
        if self.vr_compositor is None:
            raise Exception('unable to create compositor')
        self.vr_framebuffers[0].init_gl()
        self.vr_framebuffers[1].init_gl()
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self.poses = poses_t()
        self.projection_matrices = (np.asarray(matrixForOpenVRMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Left,
                                                                                                        znear, zfar))),
                                    np.asarray(matrixForOpenVRMatrix(self.vr_system.getProjectionMatrix(openvr.Eye_Right,
                                                                                                        znear, zfar))))
        self.eye_transforms = (np.asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Left)).I),
                               np.asarray(matrixForOpenVRMatrix(self.vr_system.getEyeToHeadTransform(openvr.Eye_Right)).I))
        self.view_matrices = (np.empty((4,4), dtype=np.float32),
                              np.empty((4,4), dtype=np.float32))
        self.hmd_matrix = np.eye(4, dtype=np.float32)
        self.vr_event = openvr.VREvent_t()
        self._controller_indices = []
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
                self._controller_indices.append(i)
    def update_projection_matrix(self):
        pass
    def init_gl(self, clear_color=(0.0, 0.0, 0.0, 0.0)):
        gl.glClearColor(*clear_color)
        gl.glEnable(gl.GL_DEPTH_TEST)
        #gl.glViewport(0, 0, self.window_size[0], self.window_size[1])
    @contextmanager
    def render(self, meshes=None):
        self.vr_compositor.waitGetPoses(self.poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
        hmd_pose = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not hmd_pose.bPoseIsValid:
            yield None
            return
        hmd_34 = np.ctypeslib.as_array(cast(hmd_pose.mDeviceToAbsoluteTracking.m, c_float_p),
                                       shape=(3,4))
        poses = [hmd_34]
        velocities = [np.ctypeslib.as_array(hmd_pose.vVelocity.v)]
        angular_velocities = [np.ctypeslib.as_array(hmd_pose.vAngularVelocity.v)]
        self.hmd_matrix[:,:3] = hmd_34.T
        view = np.linalg.inv(self.hmd_matrix)
        for i in self._controller_indices:
            controller_pose = self.poses[i]
            if controller_pose.bPoseIsValid:
                pose_34 = np.ctypeslib.as_array(cast(controller_pose.mDeviceToAbsoluteTracking.m, c_float_p),
                                                shape=(3,4))
                poses.append(pose_34)
                velocities.append(np.ctypeslib.as_array(controller_pose.vVelocity.v))
                angular_velocities.append(np.ctypeslib.as_array(controller_pose.vAngularVelocity.v))
        for eye in (0,1):
            view.dot(self.eye_transforms[eye], out=self.view_matrices[eye])

        frame_data = {
            'hmd_pose': hmd_34,
            'hmd_velocity': velocities[0],
            'hmd_angular_velocity': angular_velocities[0],
            'controller_poses': poses[1:],
            'controller_velocities': velocities[1:],
            'controller_angular_velocities': angular_velocities[1:],
            'view_matrices': self.view_matrices,
            'projection_matrices': self.projection_matrices,
        }

        yield frame_data

        gl.glViewport(0, 0, self.vr_framebuffers[0].width, self.vr_framebuffers[0].height)
        for eye in (0,1):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.vr_framebuffers[eye].fb)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            frame_data['view_matrix'] = self.view_matrices[eye]
            frame_data['projection_matrix'] = self.projection_matrices[eye]
            if meshes is not None:
                for mesh in meshes:
                    mesh.draw(**frame_data)
        #self.vr_compositor.submit(openvr.Eye_Left, self.vr_framebuffers[0].texture)
        self.vr_framebuffers[0].submit(openvr.Eye_Left)
        #self.vr_compositor.submit(openvr.Eye_Right, self.vr_framebuffers[1].texture)
        self.vr_framebuffers[1].submit(openvr.Eye_Right)
        # mirror left eye framebuffer to screen:
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        # gl.glDrawBuffer(gl.GL_BACK)
        gl.glBlitNamedFramebuffer(self.vr_framebuffers[0].fb, 0,
                                  0, 0, self.vr_framebuffers[0].width, self.vr_framebuffers[0].height,
                                  0, 0, self.window_size[0], self.window_size[1],
                                  gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    def process_input(self, button_press_callbacks=None):
        for i in self._controller_indices:
            got_state, state = self.vr_system.getControllerState(i, 1)
            if got_state and state.rAxis[1].x > 0.05:
                self.vr_system.triggerHapticPulse(i, 0, int(3200 * state.rAxis[1].x))
        if self.vr_system.pollNextEvent(self.vr_event):
            if button_press_callbacks and self.vr_event.eventType == openvr.VREvent_ButtonPress:
                button = self.vr_event.data.controller.button
                if button in button_press_callbacks:
                    button_press_callbacks[button]()
            elif self.vr_event.eventType == openvr.VREvent_ButtonUnpress:
                pass
    def shutdown(self):
        openvr.shutdown()
