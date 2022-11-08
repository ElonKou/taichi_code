# created by: elonkou@ktime.cc
# 2022.11.08
# reference code: taichi https://www.bilibili.com/video/BV1at4y1K76P

import math
import taichi as ti

ti.init(arch=ti.gpu)

# screen
# res = (960, 540)
res = (1920, 1080)
pixels = ti.Vector.field(n=3, dtype=float, shape=res)

# camera
fov = 120
tan_half_fov = math.tan(fov / 360 * math.pi)
z_near, z_far, grid_size = 200, 3200, 120
N = (res[0]//grid_size, res[1]//grid_size, (z_far - z_near) // grid_size)
pos = ti.Vector.field(n=3, dtype=float, shape=N)
color = ti.Vector.field(n=3, dtype=float, shape=N)
vel = ti.Vector.field(n=3, dtype=float, shape=())


# motion
fps = 60.0
dt = 1.0 / fps
t = 0.0


@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.func
def spawn(I):
    pos[I] = (I + rand3()) * grid_size
    pos[I].z += z_near
    color[I] = rand3() + ti.Vector([1.0, 2.0, 3.0])


@ti.kernel
def step():
    for I in ti.grouped(pos):
        pos[I] += vel[None] * dt
        if pos[I].z < z_near:
            pos[I].z += z_far - z_near
            pos[I].x = vel[None].z * ti.cos(pos[I].x)


@ti.kernel
def init():
    for I in ti.grouped(pos):
        spawn(I)


@ti.func
def project(pos_3d):
    center = ti.Vector(res) / 2
    w = tan_half_fov * pos_3d.z
    res_pos = (ti.Vector([pos_3d.x, pos_3d.y]) - center) / w
    screen_pos = res_pos * res[1] + center
    return screen_pos


@ti.func
def draw_particle(c, rad, col):
    x_range = (ti.max(int(c.x - rad - 1), 0), ti.min(int(c.x + rad + 1), res[0]))
    y_range = (ti.max(int(c.y - rad - 1), 0), ti.min(int(c.y + rad + 1), res[1]))
    for I in ti.grouped(ti.ndrange(x_range, y_range)):
        dist = (I - c).norm()
        if dist < rad:
            pixels[I] = col


@ti.func
def draw_line(p1, p2, rad, col):
    x_range = (ti.max(int(ti.min(p1.x, p2.x) - rad - 1.0), 0), ti.min(int(ti.max(p1.x, p2.x) + rad + 1.0), res[0]))
    y_range = (ti.max(int(ti.min(p1.y, p2.y) - rad - 1.0), 0), ti.min(int(ti.max(p1.y, p2.y) + rad + 1.0), res[1]))
    for I in ti.grouped(ti.ndrange(x_range, y_range)):
        p1I = I - p1
        p2I = I - p2
        p1p2_norm = (p2 - p1).normalized()
        p = p1 + p1I.dot(p1p2_norm) * p1p2_norm  # point on th line of [p1, p2]
        dist_p = (I - p).norm()
        dist_p1p2 = ti.min(p1I.norm(), p2I.norm())
        dist_line = dist_p

        if (p1I.dot(p1p2_norm) < 0) or (p2I.dot(p1p2_norm) > 0):
            dist_line = dist_p1p2

        if dist_line < rad:
            pixels[I] = col

# reference: https://iquilezles.org/articles/distfunctions2d/
# float sdStar5(in vec2 p, in float r, in float rf)
# {
#     const vec2 k1 = vec2(0.809016994375, -0.587785252292);
#     const vec2 k2 = vec2(-k1.x,k1.y);
#     p.x = abs(p.x);
#     p -= 2.0*max(dot(k1,p),0.0)*k1;
#     p -= 2.0*max(dot(k2,p),0.0)*k2;
#     p.x = abs(p.x);
#     p.y -= r;
#     vec2 ba = rf*vec2(-k1.y,k1.x) - vec2(0,1);
#     float h = clamp( dot(p,ba)/dot(ba,ba), 0.0, r );
#     return length(p-ba*h) * sign(p.y*ba.x-p.x*ba.y);
# }

@ti.func
def draw_star(c, rad, col):
    rf = 0.6
    x_range = (ti.max(int(c.x - rad - 1), 0), ti.min(int(c.x + rad + 1), res[0]))
    y_range = (ti.max(int(c.y - rad - 1), 0), ti.min(int(c.y + rad + 1), res[1]))
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    rad = rad / width
    for I in ti.grouped(ti.ndrange(x_range, y_range)):
        # convert image space to viewport.
        p = I - ti.Vector([(x_range[0] + x_range[1]) * 0.5, (y_range[0] + y_range[1]) * 0.5])
        p[0] = p[0] / width  # [-0.5, 0.5]
        p[1] = p[1] / height  # [-0.5, 0.5]

        # generate distance field of 2D-stars
        k1 = ti.Vector([0.809016994375, -0.587785252292])
        k2 = ti.Vector([-k1[0], k1[1]])
        p[0] = ti.abs(p[0])
        p -= 2.0 * ti.max(k1.dot(p), 0.0) * k1
        p -= 2.0 * ti.max(k2.dot(p), 0.0) * k2
        p[0] = ti.abs(p[0])
        p[1] -= rad

        ba = rf * ti.Vector([-k1[1], k1[0]]) - ti.Vector([0, 1])
        h = p.dot(ba) / ba.dot(ba)
        h = ti.max(h, rad)
        h = ti.min(h, 0.0)

        v = p[1] * ba[0] - p[0] * ba[1]
        if v > 0:
            v = 1
        elif v < 0.0:
            v = -1.0

        dist = (p - ba * h).norm() * v # diatance field of 2D-stars

        # render sdf.
        if dist < 0.0:
            pixels[I] = col



@ti.kernel
def paint():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.8, 0.8, 0.8]) * pixels[i, j]

    # draw_star(ti.Vector([res[0]//2, res[1]//2]), 300.0, ti.Vector([0.1, 0.65, 0.7]))
    
    for I in ti.grouped(pos):
        rad = 10.0 * (1.0 - pos[I].z / z_far)**2
        col = color[I] * (1.0 - pos[I].z / z_far + 0.1) ** 0.5

        dist = ((pos[I].x / res[0] - 0.5)**2 + (pos[I].y / res[1] - 0.5)**2)**0.2  # distance from pos to center.
        col = col * (1.0 - dist)

        cur_p, pre_p = pos[I], pos[I] - vel[None] * dt

        p1 = project(cur_p)
        p2 = project(pre_p)
        # draw_line(p1, p2, rad, col)
        draw_star(p1, rad, col)


if __name__ == "__main__":
    init()
    gui = ti.GUI("1024", res, fast_gui=True)
    while gui.running:
        t += dt
        vel[None].z = -1000 * (1.2 + math.cos(t * math.pi))
        step()
        paint()
        gui.set_image(pixels)
        gui.show()
