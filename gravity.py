import math
import taichi as ti

ti.init(arch = ti.gpu)

# screen space variables
res = (1920, 1280)
pixels = ti.Vector.field(3, dtype = float, shape = res)

# world space variables
grid_spacing = 100
z_near, z_far = 200, 3200
N = (res[0] // grid_spacing, res[1] // grid_spacing, (z_far - z_near) // grid_spacing)
pos = ti.Vector.field(3, dtype = float, shape = N)
vel = ti.Vector.field(3, dtype = float, shape = N)
force = ti.Vector.field(3, dtype = float, shape = N)
color = ti.Vector.field(3, dtype = float, shape = N)
fps = 60
dt = 1.0 / fps

#camera related variables
fov = 120
tan_half_fov = math.tan(fov / 360 * math.pi)

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def spawn(I):
    pos[I] = (I + rand3()) * grid_spacing
    pos[I].z += z_near
    force[I] = ti.Vector.zero(float, 3, 1)
    vel[I] = (rand3()-0.5) * 50
    color[I] = rand3() + ti.Vector([1, 2, 3])

@ti.kernel
def init():
    for I in ti.grouped(pos):
        spawn(I)

@ ti.kernel
def compute_force():
    for I in ti.grouped(pos):
        force[I].fill(0)
    for I in ti.grouped(pos):
        for J in ti.grouped(ti.ndrange((0, N[0]), (0, N[1]), (0, N[2]))):
            if (I-J).any() != 0:
                r = pos[J] - pos[I]
                diff = r.norm(1e-2)
                force[I] +=  1000 * r / diff ** 3 # GMr/r^2 * r.normalized()

@ti.kernel
def step():
    for I in ti.grouped(pos):
        vel[I] += force[I] * dt # assume unit mass
        pos[I] += vel[I] * dt
        if not 0 < pos[I].x < res[0] or not 0 < pos[I].y < res[1] or not z_near < pos[I].z < z_far:
            spawn(I) # respawn when out of box

# paint related functions
@ti.func
def draw_line(p1, p2, rad, col):
    x_range = (ti.max(ti.min(int(p1.x - rad - 1), int(p2.x - rad - 1)), 0), ti.min(ti.max(int(p1.x + rad + 1), int(p2.x + rad + 1)), res[0] - 1))
    y_range = (ti.max(ti.min(int(p1.y - rad - 1), int(p2.y - rad - 1)), 0), ti.min(ti.max(int(p1.y + rad + 1), int(p2.y + rad + 1)), res[1] - 1))
    for I in ti.grouped(ti.ndrange(x_range, y_range)):
        p1I, p1p2_normalized = I - p1, (p2 - p1).normalized()
        p1p = p1I.dot(p1p2_normalized) * p1p2_normalized
        dist = ti.min((I - p1).norm(), (I - p2).norm())
        if 0 < p1I.dot(p1p2_normalized) < (p2 - p1).norm():
            dist = (p1I - p1p).norm()
        if dist < rad:
            pixels[I] += col * (1-dist/rad)**2

# camera
@ti.func
def project(pos):
    # assume camera is located at center of view space
    center = ti.Vector(res) / 2
    w = tan_half_fov * pos.z
    rel_pos = (ti.Vector([pos.x, pos.y]) - center) / w
    return rel_pos * res[0] + center

@ti.kernel
def paint():
    for I in ti.grouped(pos):
        p_current, p_previous = project(pos[I]), project(pos[I] - vel[I] * dt)
        rad = 5.0 * (1.0 - pos[I].z/z_far)**0.5
        col = color[I] * (1.0 - pos[I].z/z_far)**0.5 # depth fade
        col *= 1.0 - (2 *((pos[I].x/res[0]-0.5) ** 2 + (pos[I].y/res[1]-0.5) ** 2)) ** 0.2 # radial fade
        draw_line(p_current, p_previous, rad, col)

@ti.kernel
def clear_canvas():
    for I in ti.grouped(pixels):
        pixels[I] *= 0.5

init()
# ggui = ti.ui.Window("Interstellar", res, vsync=True)
gui = ti.GUI('Interstellar', res, fast_gui = True) # use GUI instead of GGUI if you are using metal backend
t = 0.0
while gui.running:
    # update vel and take a step
    compute_force()
    step()
    t += dt
    # draw something
    clear_canvas()
    paint()
    gui.set_image(pixels)
    gui.show()