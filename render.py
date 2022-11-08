# !/bin/env python3
# -*- coding: utf-8 -*-

import taichi as ti

ti.init(arch=ti.vulkan, debug=True)

pixels = ti.Vector.field(1, ti.f32)
# hierarchical layout
hl = ti.root.dense(ti.i, 400).dense(ti.j, 400)
# flat layout
# fl = ti.root.dense(ti.i, 400).dense(ti.j, 400)

hl.place(pixels)


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = [0.0]


if __name__ == '__main__':
    window = ti.ui.Window('Perlin Noise', (400, 400))
    canvas = window.get_canvas()
    while window.running:
        render()

        canvas.set_image(pixels)
        window.show()
