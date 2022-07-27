import taichi as ti
import numpy as np
import sys
from taichi.math import *
import os
from PIL import Image, ImageFilter

ti.init(arch=ti.vulkan)

DIRECTORY_PATH = os.path.dirname(__file__)
HOURGLASS_IMAGE_FILE = "hg.png"
BACKGROUND_IMAGE_FILE = "bw.jpg"
DEVICE_WIDTH = 1440
DEVICE_HEIGHT = 3216
RESOLUTION_FACTOR = 4
resolution = (DEVICE_WIDTH//RESOLUTION_FACTOR, DEVICE_HEIGHT//RESOLUTION_FACTOR)

PI = 3.1415926535
grav = 9.8
ColRestitution = 0.8

quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 3000 * quality ** 2, 100 * quality
R = ivec2(n_grid, n_grid)

dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5/ quality
p_vol = (dx * 0.5) ** 2
E, nu = 5.0e3, 0.3 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

gravity = ti.Vector.field(2, dtype=float, shape=())
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
F = ti.Matrix.field(2, 2, float, n_particles)
m = ti.field(float, n_particles)
J = ti.field(float, n_particles)
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

hourglass_img = ti.Vector.field(4, ti.f32, shape=resolution)
background_img = ti.Vector.field(4, ti.f32, shape=resolution)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
grid_c = ti.Vector.ndarray(4, dtype=float, shape=resolution)
grid_c_field = ti.Vector.field(4, dtype=float, shape=resolution)

# loads image assets
def load_assets():
    hourglass_image = Image.open(os.path.join(DIRECTORY_PATH, HOURGLASS_IMAGE_FILE))
    background_image = Image.open(os.path.join(DIRECTORY_PATH, BACKGROUND_IMAGE_FILE))

    hourglass_image_resized = hourglass_image.resize(resolution).transpose(
                Image.ROTATE_270)
    background_image_resized = background_image.resize(resolution).transpose(
                Image.ROTATE_270)
    hourglass_image_resized.save('hourglass.png')
    background_image_resized.save('background.png')

    # loads images as np ndarray
    hourglass_img_np = np.asarray(hourglass_image_resized).astype(np.float32) / 255.0
    background_img_np = np.asarray(background_image_resized).astype(np.float32) / 255.0

    hourglass_img.from_numpy(hourglass_img_np)
    background_img.from_numpy(background_img_np)

@ti.kernel
def copy_to_field(arr: ti.types.ndarray()):
    for I in ti.grouped(arr):
        grid_c_field[I] = arr[I]

@ti.kernel
def reset():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

@ti.kernel
def P2G():
    for p in range(n_particles):  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] # deformation gradient update
        # h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))
        h = 1.0
        la = lambda_0 * h
        mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                 ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        affine = stress + m[p] * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (m[p] * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * m[p]

@ti.func
def normal(p):
    eps = 0.01
    h = vec2(eps,0)
    return normalize( vec2(border(p + h.xy) - border(p - h.xy),
                           border(p + h.yx) - border(p - h.yx)))

@ti.kernel
def grid_operator():
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None]* 10  # gravity
            # if i < 3 and grid_v[i, j][0] < 0:
            #     grid_v[i, j][0] = 0  # Boundary conditions
            # if i > n_grid - 3 and grid_v[i, j][0] > 0: 
            #     grid_v[i, j][0] = 0
            # if j < 3 and grid_v[i, j][1] < 0: 
            #     grid_v[i, j][1] = 0
            # if j > n_grid - 3 and grid_v[i, j][1] > 0: 
            #     grid_v[i, j][1] = 0
            v = grid_v[i, j]
            P = vec2(i, j)
            d = border(P)
            n = normal(P)
            if (d > 0.):
                grid_v[i, j] -= n * d * ColRestitution
                grid_v[i, j] *= 0.9

@ti.func
def border(p):
    width = 2.39092
    height = 5.3316
    h0 = 1.0731/height*R.y*0.5

    center0 = p
    center0.x = center0.x - R.x*0.5 
    center0.y = center0.y - R.y*0.20/2.0

    boundary0 = sdCutDisk(center0,  R.x*0.75/2.0, R.y*0.1/2.0)

    center1 = p 
    center1.x = center1.x - R.x*0.5     

    center1.y = center1.y - R.y*1.81/2    
    center1 = rotate(center1, 3.14)

    boundary1 = sdCutDisk(center1,  R.x*0.75/2.0, R.y*0.1/2.0)
    center2 = p 
    center2.x = center2.x - R.x*0.5 
    center2.y = center2.y - R.y*0.5
    
    boundary2 = sdBox(center2,  vec2(R.x*0.06/2.0, R.y*0.18/2.0))
    boundary = merge(merge(boundary0, boundary2), boundary1)
    return boundary

@ti.func
def rotate(v, a):
    s = ti.sin(a)
    c = ti.cos(a)
    m = mat2([[c, s],[-s, c]])
    return m @ v
   
@ti.func 
def sdBox(p, b):
    d = abs(p) - b
    return (max(d, 0.0).norm() + min(max(d.x, d.y), 0.0))

@ti.func 
def sdCutDisk(p, r, h):
    w = ti.sqrt(r*r-h*h)
    p.x = abs(p.x)
    s = max( (h-r)*p.x*p.x+w*w*(h+r-2.0*p.y), h*p.x-w*p.y )
    res = 0.0
    if s < 0.0:
        res = p.norm() - r
    else:
        if p.x < w:
            res = h - p.y
        else:
            res = (p - vec2(w, h)).norm()
    return res 

@ti.func
def merge(d1, d2):
    return min(d1, d2)

@ti.func 
def intersection(d1, d2):
    return max(d1, d2)

@ti.kernel
def G2P():
    for p in range(n_particles):  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = (ti.Vector([i, j]).cast(float) - fx)
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p] 

@ti.kernel
def add_front(grid_c: ti.types.ndarray()):
    for i, j in grid_c:
        if (background_img[i, j][2] < 0.1):
            grid_c[i, j][0] = background_img[i, j][0]
            grid_c[i, j][1] = background_img[i, j][1]
            grid_c[i, j][2] = background_img[i, j][2]
            grid_c[i, j][3] = background_img[i, j][3]

@ti.kernel
def init():
    gravity[None] = [0, -grav]
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.50, ti.random() * 0.2 + 0.6]
        v[i] = ti.Vector([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        m[i] = p_vol * 1.0
        Jp[i] = 1

@ti.func
def sdCircle(p, r):
    return (p).norm() -r

@ti.func
def smoothFilter(d):
    v = 2000. / resolution[1]
    return smoothstep(v, -v, d)

@ti.kernel
def smooth(grid_c: ti.types.ndarray()):
    for i, j in grid_c:
        fragCoord = vec2(i, j)
        p = fragCoord
        col = vec3(0.0)
        col = mix(col, vec3(0.0), smoothFilter(border(p)))
        sminAcc = 0.0            
        a = 0.5
        for k in range(n_particles):
            pp = x[k]*resolution
            sminAcc += 2**(-a*sdCircle(p - pp,8.0))
        col = mix(col, vec3(0.0, 0.0, 1.0), smoothFilter(-log2(sminAcc)/2.0))
        grid_c[i,j] = vec4(col, 1.0)
        if (grid_c[i, j][2] < 0.8):
            grid_c[i, j] = 0.5*hourglass_img[i, j]+vec4(0.0, 0.0, 0.0,0.5)

def main():
    init()
    gui = ti.GUI('Hourglass', res=resolution, background_color=0x112F41)
    window = ti.ui.Window('Hourglass', resolution)
    canvas = window.get_canvas()
    while window.running:
        # if canvas.get_event(ti.GUI.PRESS):
        #     if gui.event.key == 'r': reset()
        #     elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
        # if gui.event is not None: gravity[None] = [0, 0]  # if had any event
        # if gui.is_pressed(ti.GUI.LEFT, 'a'): gravity[None][0] = -grav
        # if gui.is_pressed(ti.GUI.RIGHT, 'd'): gravity[None][0] = grav
        # if gui.is_pressed(ti.GUI.UP, 'w'): gravity[None][1] = grav
        # if gui.is_pressed(ti.GUI.DOWN, 's'): gravity[None][1] = -grav

        for s in range(int(2e-3 // dt)):
            reset()
            P2G()
            grid_operator()
            G2P()

        smooth(grid_c)
        # canvas.circles(x, color=(1.0, 0.5, 0.5), radius=0.01)        

        add_front(grid_c)
        copy_to_field(grid_c)
        canvas.set_image(grid_c_field)
        window.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk

def aot():
    m = ti.aot.Module(ti.vulkan)
    m.add_kernel(init)
    m.add_kernel(reset)
    m.add_kernel(P2G)
    m.add_kernel(grid_operator)
    m.add_kernel(G2P)
    m.add_kernel(smooth,
                template_args={
                    'grid_c': grid_c                
                    })
    m.add_kernel(add_front,
                template_args={
                    'grid_c': grid_c                
                    })
    m.save('.', 'hourglass')


if __name__ == '__main__':
    load_assets()
    # for arg in sys.argv:
    #     print(arg)
    main()
    aot()
