import math

# EXACT replica of the current C++ generator (parallel transport + twist +
# canonical curve R=2.2/r=0.9/tube 0.10, radius=1) + software rasterizer.
R, r_tube, p, q = 2.2, 0.9, 2, 3
tubeR = 0.10
tubeSegs, ringSegs = 256, 32

def curve(t):
    cr = R + r_tube * math.cos(q*t)
    return (cr*math.cos(p*t), cr*math.sin(p*t), r_tube*math.sin(q*t))

def tang(t):
    a, b = curve(t), curve(t+0.01)
    d = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    l = math.sqrt(sum(x*x for x in d))
    return tuple(x/l for x in d)

def sub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    l = math.sqrt(sum(x*x for x in a))
    return tuple(x/l for x in a) if l > 1e-12 else (0,0,0)
def dot(a,b): return sum(x*y for x,y in zip(a,b))

t0 = tang(0.0)
seed = (0,1,0)
if abs(dot(seed,t0)) > 0.9: seed = (1,0,0)
prevB = norm(cross(t0, norm(cross(seed, t0))))

transported = prevB
for i in range(tubeSegs):
    T = tang(i/tubeSegs*2*math.pi)
    transported = norm(sub(transported, tuple(T[k]*dot(T,transported) for k in range(3))))
seam = math.atan2(dot(cross(prevB, transported), t0), dot(prevB, transported))
twist = -seam/tubeSegs

verts, norms = [], []
b = prevB
for i in range(tubeSegs+1):
    t = i/tubeSegs*2*math.pi
    p1 = curve(t); T = tang(t)
    b = norm(sub(b, tuple(T[k]*dot(T,b) for k in range(3))))
    a_ = twist*i
    side = norm(cross(T, b))
    b = norm(tuple(b[k]*math.cos(a_) + side[k]*math.sin(a_) for k in range(3)))
    n = cross(b, T)
    for j in range(ringSegs+1):
        v = j/ringSegs*2*math.pi
        cx, cy = -tubeR*math.cos(v), tubeR*math.sin(v)
        pos = tuple(p1[k] + cx*n[k] + cy*b[k] for k in range(3))
        verts.append(pos)
        nr = norm(sub(pos, p1))
        norms.append(nr)

def idx(i,j): return i*(ringSegs+1) + (j % (ringSegs+1))
tris = []
for i in range(tubeSegs):
    for j in range(ringSegs):
        a, b2, c, d = idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1)
        tris.append((a,b2,c)); tris.append((a,c,d))

# sanity 1: degenerate triangles?
deg = sum(1 for t in tris if verts[t[0]]==verts[t[1]] or verts[t[1]]==verts[t[2]] or verts[t[0]]==verts[t[2]])
print(f"triangles: {len(tris)}, degenerate: {deg}")

# sanity 2: per-triangle outward consistency
bad = 0
for ti, t in enumerate(tris):
    A,B,C = verts[t[0]], verts[t[1]], verts[t[2]]
    fn = cross(sub(B,A), sub(C,A))
    # radial at centroid: nearest curve point (search)
    cx = (A[0]+B[0]+C[0])/3; cy = (A[1]+B[1]+C[1])/3; cz = (A[2]+B[2]+C[2])/3
    best = min(range(0, 360), key=lambda k: math.dist((cx,cy,cz), curve(k/360*2*math.pi)))
    pc = curve(best/360*2*math.pi)
    outward = sub((cx,cy,cz), pc)
    if dot(fn, outward) <= 0: bad += 1
print(f"inward-facing triangles: {bad}/{len(tris)}")

# software render: camera at yaw=-60 pitch=-40 dist=10 (orbit: target - fwd*d)
import struct
yaw, pitch, dist = math.radians(-60), math.radians(-40), 10.0
target = (0,0,0)
cp_, sp = math.cos(pitch), math.sin(pitch)
fwd = (cp_*math.sin(yaw), sp, -cp_*math.cos(yaw))
eye = tuple(target[k] - dist*fwd[k] for k in range(3))
# view basis
zaxis = norm(tuple(-eye[k] + target[k] for k in range(3)))  # target - eye normalized... careful
# proper lookAt RH: z = normalize(eye - target), x = normalize(cross(up,z)), y = cross(z,x)
up = (0,1,0)
zax = norm(sub(eye, target))
xax = norm(cross(up, zax))
yax = cross(zax, xax)
fovy = math.radians(60)
W, H = 320, 240
zbuf = [999.0]*(W*H)
shade = [0.0]*(W*H)
def project(v):
    d = sub(v, eye)
    vx, vy, vz = dot(d,xax), dot(d,yax), dot(d,zax)
    if vz < 0.01: return None
    f = 1.0/math.tan(fovy/2)
    sx = (vx*f*0.5*W + W/2) * (W/ (W*0.5*2)) # keep simple: sx in [0,W)
    sx = vx/vz * f * W/2 + W/2
    sy = -vy/vz * f * H/2 + H/2
    return sx, sy, vz

for t in tris:
    P = [project(verts[k]) for k in t]
    if any(pp is None for pp in P): continue
    x0 = max(0, int(min(pp[0] for pp in P))); x1 = min(W-1, int(max(pp[0] for pp in P)))
    y0 = max(0, int(min(pp[1] for pp in P))); y1 = min(H-1, int(max(pp[1] for pp in P)))
    ax, ay, az = P[0]; bx, by, bz = P[1]; cx2, cy2, cz2 = P[2]
    for yy in range(y0, y1+1):
        for xx in range(x0, x1+1):
            px_, py_ = xx+0.5, yy+0.5
            d = (bx-ax)*(cy2-ay) - (by-ay)*(cx2-ax)
            if abs(d) < 1e-9: continue
            w0 = ((bx-ax)*(py_-ay) - (by-ay)*(px_-ax)) / d
            w1 = ((py_-ay)*(cx2-ax) - (px_-ax)*(cy2-ay)) / d
            w2 = 1 - w0 - w1
            if w0 < 0 or w1 < 0 or w2 < 0: continue
            z = w0*az + w1*bz + w2*cz2
            if z < zbuf[yy*W+xx]:
                zbuf[yy*W+xx] = z
                n = tuple(w0*norms[t[0]][k] + w1*norms[t[1]][k] + w2*norms[t[2]][k] for k in range(3))
                n = norm(n)
                l = norm((0.35,0.8,0.45))
                shade[yy*W+xx] = max(0.0, dot(n,l))

# ASCII out
cols, rows_n = 76, 26
for ry in range(rows_n):
    line = ''
    for rx in range(cols):
        x0 = rx*W//cols; x1 = (rx+1)*W//cols
        y0 = ry*H//rows_n; y1 = (ry+1)*H//rows_n
        best = max(shade[y*W+x] for y in range(y0,y1) for x in range(x0,x1)) if any(zbuf[y*W+x]<999 for y in range(y0,y1) for x in range(x0,x1)) else -1
        if best < 0: line += '.'
        else:
            line += ' ' if best > 0.75 else ('#' if best > 0.45 else ('+' if best > 0.2 else '*'))
    print(line)
