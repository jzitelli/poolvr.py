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
        yield (poses, velocities, angular_velocities)
        gl.glViewport(0, 0, self.vr_framebuffers[0].width, self.vr_framebuffers[0].height)
        for eye in (0,1):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.vr_framebuffers[eye].fb)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            view.dot(self.eye_transforms[eye], out=self.view_matrices[eye])
            if meshes is not None:
                for mesh in meshes:
                    mesh.draw(projection=self.projection_matrices[eye],
                              view=self.view_matrices[eye])
        self.vr_compositor.submit(openvr.Eye_Left, self.vr_framebuffers[0].texture)
        self.vr_compositor.submit(openvr.Eye_Right, self.vr_framebuffers[1].texture)
        # mirror left eye framebuffer to screen:
        gl.glBlitNamedFramebuffer(self.vr_framebuffers[0].fb, 0,
                                  0, 0, self.vr_framebuffers[0].width, self.vr_framebuffers[0].height,
                                  0, 0, self.window_size[0], self.window_size[1],
                                  gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    def process_input(self):
        for i in self._controller_indices:
            got_state, state = self.vr_system.getControllerState(i, 1)
            if got_state and state.rAxis[1].x > 0.05:
                self.vr_system.triggerHapticPulse(i, 0, int(3200 * state.rAxis[1].x))
        if self.vr_system.pollNextEvent(self.vr_event):
            if self.vr_event.eventType == openvr.VREvent_ButtonPress:
                pass
            elif self.vr_event.eventType == openvr.VREvent_ButtonUnpress:
                pass
    def shutdown(self):
        openvr.shutdown()
