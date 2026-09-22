import glm
import numpy as np
import OpenGL.GL.shaders as shaders
from OpenGL.GL import *


class Camera:
    def __init__(self, h, w):
        self.znear = 0.00001
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = np.pi / 2
        self.position = np.array([0.0, 0.0, 0.5]).astype(np.float32)
        self.target = np.array([0.0, -1.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, -1.0, 0.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True

        self.last_x = 0
        self.last_y = 0
        self.first_mouse = True

        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False

        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.0

        self.view_matrix = np.eye(4).astype(np.float32)

    def get_view_matrix(self):
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_project_matrix(self):
        project_mat = glm.perspective(self.fovy, self.w / self.h, self.znear, self.zfar)
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def update_resolution(self, height, width):
        self.h = height
        self.w = width
        self.is_intrin_dirty = True


def load_shaders(vs, fs):
    vertex_shader = open(vs, "r").read()
    fragment_shader = open(fs, "r").read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = glGenBuffers(1)
            buffer_ids[i] = b
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
        length = value.shape[-1]
        pos = glGetAttribLocation(program, key)
        glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vao, buffer_ids


def set_storage_buffer_data(
    program, value: np.ndarray, bind_idx, vao=None, buffer_id=None
):
    glUseProgram(program)
    if vao is not None:
        glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW
    )
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id


def set_faces_tovao(vao, faces: np.ndarray):
    # faces
    glBindVertexArray(vao)
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    return element_buffer


def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 1, GL_FALSE, content.astype(np.float32)
    )


def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name),
        content,
    )


def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, name), content)


def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name), contents[0], contents[1], contents[2]
    )
