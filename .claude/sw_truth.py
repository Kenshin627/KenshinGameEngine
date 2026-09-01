import math, struct

# ---- exact generator replica (current C++: R=2.2 r=0.9 tube 0.16, per-step twist)
R, r_tube, p_, q_ = 2.2, 0.9, 2, 3
tubeR, tubeSegs, ringSegs = 0.16, 256, 32
def curve(t):
    cr = R + r_tube*math.cos(q_*t)
    return (cr*math.cos(p_*t), cr*math.sin(p_*t), r_tube*math.sin(q_*t))
def sub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    l = math.sqrt(sum(x*x for x in a))
    return tuple(x/l for x in a) if l > 1e-12 else (0,0,0)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def tang(t):
    a, b = curve(t), curve(t+0.01)
    return norm(sub(b,a))
t0 = tang(0.0)
seed = (0,1,0)
if abs(dot(seed,t0)) > 0.9: seed = (1,0,0)
prevB = norm(cross(t0, norm(cross(seed, t0))))
tr = prevB
for i in range(tubeSegs):
    T = tang(i/tubeSegs*2*math.pi)
    tr = norm(sub(tr, tuple(T[k]*dot(T,tr) for k in range(3))))
seam = math.atan2(dot(cross(prevB, tr), t0), dot(prevB, tr))
tw = -seam/tubeSegs
verts, norms = [], []
b = prevB
for i in range(tubeSegs+1):
    t = i/tubeSegs*2*math.pi
    p1 = curve(t); T = tang(t)
    b = norm(sub(b, tuple(T[k]*dot(T,b) for k in range(3))))
    side = norm(cross(T, b))
    b = norm(tuple(b[k]*math.cos(tw) + side[k]*math.sin(tw) for k in range(3)))
    n = cross(b, T)
    for j in range(ringSegs+1):
        v = j/ringSegs*2*math.pi
        pos = tuple(p1[k] + (-tubeR*math.cos(v))*n[k] + (tubeR*math.sin(v))*b[k] for k in range(3))
        verts.append(pos)
        norms.append(norm(sub(pos, p1)))
def idx(i,j): return i*(ringSegs+1)+(j%(ringSegs+1))
tris = []
for i in range(tubeSegs):
    for j in range(ringSegs):
        a,b2,c,d = idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1)
        tris.append((a,b2,c)); tris.append((a,c,d))

# ---- ground-truth software rasterizer (orbit camera identical to engine)
def render(yawDeg, pitchDeg, dist, W=160, H=120):
    yaw, pitch = math.radians(yawDeg), math.radians(pitchDeg)
    target = (0.0, 0.0, 0.0)
    cp_, sp = math.cos(pitch), math.sin(pitch)
    fwd = (cp_*math.sin(yaw), sp, -cp_*math.cos(yaw))
    eye = tuple(target[k] - dist*fwd[k] for k in range(3))
    # engine: view = lookAtRH(eye, target, up); proj = perspectiveRH_ZO(60)
    zax = norm(sub(eye, target))
    xax = norm(cross((0,1,0), zax))
    yax = cross(zax, xax)
    f = 1.0/math.tan(math.radians(60)/2)
    zn, zf = 0.01, 1000.0
    zbuf = [float('inf')]*(W*H)
    img = [-1.0]*(W*H)
    for t in tris:
        P = []
        ok = True
        for k in t:
            d = sub(verts[k], eye)
            vx, vy, vz = dot(d,xax), dot(d,yax), dot(d,zax)
            if vz < 0.01: ok = False; break
            sx = (vx/vz)*f*0.5*W + W/2
            sy = (1-(vy/vz))*f*0.5*H + H/2  # Y down in image
            P.append((sx, sy, vz))
        if not ok: continue
        ax,ay,az = P[0]; bx,by,bz = P[1]; cx,cy,cz = P[2]
        # view-space winding: rasterized CW = front (engine frontFace CLOCKWISE
        # after Y flip => image-space CW with Y-down = front). With Y-down
        # screen coords, CCW-in-math == CW-on-screen; accept triangle if
        # signed area says front per engine convention. We accept BOTH for
        # ground truth (depth decides), matching CULL test below separately.
        x0=max(0,int(min(p[0] for p in P))); x1=min(W-1,int(max(p[0] for p in P))+1)
        y0=max(0,int(min(p[1] for p in P))); y1=min(H-1,int(max(p[1] for p in P))+1)
        det = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
        if abs(det) < 1e-9: continue
        for yy in range(y0,y1):
            for xx in range(x0,x1):
                px_, py_ = xx+0.5, yy+0.5
                w0 = ((bx-ax)*(py_-ay)-(by-ay)*(px_-ax))/det
                w1 = ((py_-ay)*(cx-ax)-(px_-ax)*(cy-ay))/det
                w2 = 1-w0-w1
                if w0 < -0.001 or w1 < -0.001 or w2 < -0.001: continue
                z = w0*az+w1*bz+w2*cz
                if z < zbuf[yy*W+xx]:
                    zbuf[yy*W+xx] = z
                    n = norm(tuple(w0*norms[t[0]][k]+w1*norms[t[1]][k]+w2*norms[t[2]][k] for k in range(3)))
                    l = norm((0.35,0.8,0.45))
                    img[yy*W+xx] = max(0.0, dot(n,l))
    return img, W, H

for yawDeg, pitchDeg, dist in [(-60,-40,8), (0,0,8), (30,20,8), (90,-10,8), (150,30,8)]:
    img, W, H = render(yawDeg, pitchDeg, dist)
    print(f"--- yaw {yawDeg} pitch {pitchDeg} dist {dist} ---")
    cols, rows_n = 64, 18
    for ry in range(rows_n):
        line = ''
        for rx in range(cols):
            x0=rx*W//cols; x1=(rx+1)*W//cols; y0=ry*H//rows_n; y1=(ry+1)*H//rows_n
            cell = [img[y*W+x] for y in range(y0,y1) for x in range(x0,x1) if img[y*W+x] >= 0]
            if not cell: line += '.'
            else:
                best = max(cell)
                line += ' ' if best > 0.7 else ('#' if best > 0.4 else ('+' if best > 0.15 else '*'))
        print(line)
