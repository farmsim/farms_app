""" OpenGL MSAA framebuffer for offscreen rendering to ImGui textures """


import OpenGL.GL as GL


class MSAAFramebuffer:
    """MSAA framebuffer that resolves to a texture for ImGui display.

    Usage:
        fb = MSAAFramebuffer(1280, 720)
        fb.bind()           # render into it
        fb.unbind()
        fb.resolve()        # blit MSAA -> texture
        fb.texture_id       # pass to imgui.image()
    """

    def __init__(self, width: int, height: int, samples: int = 4, srgb: bool = False):
        self.width = width
        self.height = height
        self.samples = samples

        internal_format = GL.GL_SRGB8_ALPHA8 if srgb else GL.GL_RGBA8

        # MSAA framebuffer (render target)
        self.msaa_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.msaa_fbo)

        self.msaa_color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.msaa_color_rb)
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, samples, GL.GL_RGBA8, width, height,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER, self.msaa_color_rb,
        )

        self.msaa_depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.msaa_depth_rb)
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, samples, GL.GL_DEPTH_COMPONENT24, width, height,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER, self.msaa_depth_rb,
        )

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("MSAA framebuffer is not complete")

        # Resolve framebuffer (texture for ImGui)
        self.resolve_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.resolve_fbo)

        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, internal_format,
            width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, self.texture_id, 0,
        )

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Resolve framebuffer is not complete")

        # Unbind
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def bind(self):
        """Bind the MSAA framebuffer for rendering."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.msaa_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def unbind(self):
        """Unbind framebuffer."""
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        except GL.GLError:
            pass
        GL.glDisable(GL.GL_DEPTH_TEST)

    def resolve(self, filter_mode=GL.GL_NEAREST):
        """Blit MSAA framebuffer to resolve texture."""
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.msaa_fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.resolve_fbo)
        GL.glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.width, self.height,
            GL.GL_COLOR_BUFFER_BIT, filter_mode,
        )
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        except GL.GLError:
            pass

    def resize(self, width: int, height: int):
        """Resize by destroying and recreating."""
        self.destroy()
        self.__init__(width, height, self.samples)

    def destroy(self):
        """Release all GL resources."""
        GL.glDeleteTextures([self.texture_id])
        GL.glDeleteRenderbuffers(1, [self.msaa_color_rb])
        GL.glDeleteRenderbuffers(1, [self.msaa_depth_rb])
        GL.glDeleteFramebuffers(1, [self.msaa_fbo])
        GL.glDeleteFramebuffers(1, [self.resolve_fbo])
