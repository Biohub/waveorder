"""Cartoon-style inverted microscope: 3D model plus flat-shaded outline renderer.

Geometry is laid out to match a reference cartoon. Units are roughly mm.
Axes: +x is the user's right, y runs front (0) to back, z is up from the table.
"""

import math
import sys
from pathlib import Path

import cadquery as cq
import numpy as np
import vtk
from PIL import Image, ImageFilter
from vtk.util import numpy_support as ns

AX = 260.0  # y position of the optical axis


def _hex(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


# palette: (light, base, shade, outline); a "top" tone is derived from light
PAL = {
    "body": ("#f3f3f3", "#d9d9d9", "#c4c4c4", "#8c8c8c"),
    "head": ("#f0f0f0", "#cecece", "#bcbcbc", "#8c8c8c"),
    "panel": ("#d2d2d2", "#cccccc", "#c0c0c0", "#8c8c8c"),
    "white": ("#fcfcfc", "#ebebeb", "#dcdcdc", "#8c8c8c"),
    "mid": ("#c4c4c4", "#adadad", "#989898", "#777777"),
    "mid2": ("#8a8a8a", "#747474", "#666666", "#555555"),
    "dark": ("#5c5c5c", "#484848", "#3b3b3b", "#2a2a2a"),
    "blue": ("#7dd3f1", "#3aaee0", "#2e97ca", "#2a7fa8"),
    "red": ("#f08a80", "#e2574c", "#c7453b", "#8f2c25"),
    "orange": ("#f9c27a", "#f39c34", "#d98424", "#9c5c12"),
    "cobalt": ("#8fb4f5", "#4b83e8", "#3a6cc9", "#27498c"),
    "green": ("#8fdc9f", "#45b85e", "#379a4d", "#226632"),
    "yellow": ("#fbe58a", "#f3cf3f", "#d9b52e", "#8f7412"),
    "lime": ("#b5e08a", "#86c94a", "#6fae38", "#47731f"),
    "sky": ("#a9d8f5", "#6cb8ea", "#559fd0", "#2f6c96"),
}


def _tones(v):
    light, base, shade, outline = (_hex(c) for c in v)
    top = tuple(min(255, c + 5) for c in light)
    return (top, light, base, shade, outline)


PAL = {k: _tones(v) for k, v in PAL.items()}
LIGHT = np.array([-0.15, -0.8, 0.58])


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from its extents."""
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def cyl_x(d, x0, x1, y, z):
    """Cylinder with its axis along x."""
    return cq.Workplane("YZ", origin=(x0, 0, 0)).center(y, z).circle(d / 2).extrude(x1 - x0)


def cyl_y(d, y0, y1, x, z):
    """Cylinder with its axis along y."""
    return cq.Workplane("ZX", origin=(0, y0, 0)).center(z, x).circle(d / 2).extrude(y1 - y0)


def cyl_z(d, z0, z1, x=0.0, y=AX):
    """Cylinder with its axis along z."""
    return cq.Workplane("XY", origin=(0, 0, z0)).center(x, y).circle(d / 2).extrude(z1 - z0)


def prism_yz(pts, x0, x1):
    """Polygon in the (y, z) plane extruded along x."""
    return cq.Workplane("YZ", origin=(x0, 0, 0)).polyline(pts).close().extrude(x1 - x0)


def fillet(wp, selector, r):
    """Fillet selected edges, leaving the solid unchanged on failure."""
    try:
        return wp.edges(selector).fillet(r)
    except Exception:
        return wp


ROUND = 0.45  # global scale on edge radii


def soft(wp, r, selector=None, scale=True):
    """Round edges, retrying with smaller radii if the kernel refuses."""
    if scale:
        r *= ROUND
    for rr in (r, r * 0.6, r * 0.3):
        try:
            e = wp.edges(selector) if selector else wp.edges()
            return e.fillet(rr)
        except Exception:
            continue
    return wp


def knurled_z(d, z0, z1, n=15, depth=1.8):
    """Disc on the optical axis with a smooth, wavy knurled rim."""
    t = np.linspace(0, 2 * math.pi, n * 12, endpoint=False)
    r = d / 2 + depth * np.cos(n * t)
    pts = [(float(a * math.cos(b)), float(a * math.sin(b))) for a, b in zip(r, t)]
    wp = cq.Workplane("XY", origin=(0, AX, z0))
    try:
        return wp.spline(pts, periodic=True).close().extrude(z1 - z0)
    except Exception:
        return wp.polyline(pts).close().extrude(z1 - z0)


def objective(band_h=5):
    """One objective along +z from the origin: (barrel, band) solids."""
    barrel = (
        cq.Workplane("XY")
        .circle(15)
        .extrude(36)
        .faces(">Z")
        .workplane()
        .circle(15)
        .workplane(offset=11)
        .circle(8)
        .loft(combine=True)
    )
    band = cq.Workplane("XY", origin=(0, 0, 22)).circle(15.8).extrude(band_h)
    return barrel, band


def nosepiece_parts():
    """Tilted nosepiece with five objectives; the active one is vertical.

    Returns
    -------
    list of tuple
        ``(name, cq.Workplane, palette_name)`` triples in model coordinates.
    """
    alpha, r = 15, 40
    out = []
    disc = cq.Workplane("XY", origin=(0, 0, -16)).circle(64).extrude(16)
    disc = soft(disc, 5)
    cone = cq.Workplane("XY", origin=(0, 0, -34)).circle(40).workplane(offset=18).circle(64).loft(combine=True)
    local = [("nosepiece", disc.union(cone), "mid2")]
    bands = ["yellow", "lime", "sky", "white", "red"]
    for k in range(5):
        barrel, band = objective()
        for nm, wp, pal in (("barrel", barrel, "white"), ("band", band, bands[k])):
            wp = wp.rotate((0, 0, 0), (0, 1, 0), alpha).translate((r, 0, 0)).rotate((0, 0, 0), (0, 0, 1), 72 * k)
            local.append((f"objective_{nm}_{k}", wp, pal))
    ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    for nm, wp, pal in local:
        wp = (
            wp.rotate((0, 0, 0), (0, 1, 0), -alpha)
            .translate((-r * ca, 0, -r * sa))
            .rotate((0, 0, 0), (0, 0, 1), -90)
            .translate((0, AX, 322))
        )
        out.append((nm, wp, pal))
    return out


def build(highlight=False):
    """Build the microscope.

    Parameters
    ----------
    highlight : bool or set of str
        Colour the Koehler controls (field diaphragm red, condenser focus
        orange, centering screws blue, aperture lever green). Pass True for
        all, or a subset of ``{"A", "B", "C", "lever"}``.

    Returns
    -------
    list of tuple
        ``(name, cq.Workplane, palette_name)`` triples. A trailing ``!`` on
        the palette name renders the part in a single flat tone; a trailing
        ``~`` shades every curved face smoothly from its normals.
    """
    P = []

    def add(name, wp, pal):
        P.append((name, wp, pal))

    def hl(color, normal, key=""):
        on = highlight is True or (highlight and key in highlight)
        return color if on else normal

    def mirror_x(fn, side):
        wp = fn()
        return wp if side > 0 else wp.mirror("YZ")

    # ---------------- base ----------------
    body = prism_yz([(0, 0), (570, 0), (548, 430), (400, 430), (400, 375), (30, 375), (30, 267)], -100, 100)
    body = soft(body, 14)
    body = body.cut(box(-20, 101, 165, 360, 200, 366))
    add("base", body, "body")
    liner = box(-20, 98, 165, 360, 200, 366).cut(box(-17, 99, 168, 357, 203, 363))
    add("recess_liner", liner, "mid")
    add("fluor_turret", soft(box(5, 90, 172, 330, 205, 262), 7), "mid2")
    add("fluor_dial", soft(cyl_x(52, 88, 96, 230, 234), 2), "dark")
    add("fluor_lever", soft(box(84, 97, 290, 318, 208, 222), 3), "white")
    P.extend(nosepiece_parts())
    for side, sx in (("R", 1), ("L", -1)):
        add(f"focus_flange_{side}", mirror_x(lambda: cyl_x(115, 100, 106, 225, 88), sx), "white")
        add(
            f"focus_knob_{side}",
            mirror_x(lambda: soft(cyl_x(92, 106, 140, 225, 88), 6).union(soft(cyl_x(56, 138, 178, 225, 88), 6)), sx),
            "dark",
        )
    add("side_button", cyl_x(44, 100, 102, 300, 150), "body")
    add("side_slot", soft(box(100, 102, 235, 300, 178, 190), 4, "|X"), "mid2")
    add("front_slot", box(-45, -10, 0, 7, 30, 48), "mid2")
    add("tool_panel", soft(box(100, 102.5, 425, 530, 215, 330), 8, "|X"), "mid2")

    # ---------------- eyepiece tube ----------------
    add("tube_shelf", soft(box(-115, 102, -15, 115, 267, 400), 12), "body")
    add("tube_panel", prism_yz([(34, 384), (98, 384), (98, 292), (72, 292)], 102, 103.5), "panel")
    add("tube_knob", soft(cyl_y(28, -35, -15, 5, 340), 4), "dark")
    add("tube_body", soft(box(-100, 100, 0, 110, 395, 580), 26), "body")
    for i, sx in enumerate((-45, 45)):
        dark = (
            soft(cq.Workplane("XY").circle(30).extrude(75), 8)
            .union(cq.Workplane("XY", origin=(0, 0, 100)).circle(23).extrude(40))
            .union(soft(cq.Workplane("XY", origin=(0, 0, 139)).circle(29).extrude(24), 6))
        )
        band = cq.Workplane("XY", origin=(0, 0, 74)).circle(25).extrude(27)
        for nm, wp, pal in (("tube", dark, "dark"), ("band", band, "white")):
            wp = wp.rotate((0, 0, 0), (1, 0, 0), 58).translate((sx, 12, 528))
            add(f"eyepiece_{nm}_{i}", wp, pal)

    # ---------------- stage and sample ----------------
    add("stage", soft(box(-130, 130, 120, 440, 383, 395), 1.2, scale=False), "dark")
    add("stage_drum", soft(cyl_y(62, 95, 150, 92, 392), 8), "dark")
    add("stage_tab", soft(box(100, 138, 392, 500, 371, 383), 4), "dark")
    add("sample", soft(box(-65, 65, AX - 55, AX + 55, 395, 417), 4), "blue")
    add("stage_rod", cyl_z(20, 135, 383, 128, 372).union(soft(cyl_z(16, 70, 86, 128, 372), 3)), "dark")
    add("stage_rod_collar", soft(cyl_z(28, 85, 136, 128, 372), 4), "white")

    # ---------------- pillar and lamp head ----------------
    add("pillar", soft(prism_yz([(400, 425), (540, 425), (390, 745), (310, 745)], -55, 55), 12), "body")
    add("head", soft(box(-95, 95, 211, 309, 712, 850), 26), "head")
    add("head_window", soft(box(-62, 22, 208, 211.5, 760, 795), 5, "|Y"), "mid2")
    add("head_slot", soft(box(-50, 5, 208, 211.5, 728, 737), 3, "|Y"), "mid")
    add("lamphouse", soft(cyl_y(84, 305, 350, 30, 800), 10), "dark")
    add("neck", cyl_z(52, 664, 714), "mid")

    # ---------------- Koehler controls ----------------
    add("A_field_diaphragm_ring", knurled_z(84, 652, 664), hl("red~", "dark~", "A"))
    add("focus_block", soft(box(-42, 42, 310, 345, 600, 712), 8), "body")
    for side, sx in (("R", 1), ("L", -1)):
        add(f"focus_stem_{side}", mirror_x(lambda: cyl_x(14, 40, 62, 328, 650), sx), "white")
        add(
            f"B_condenser_focus_knob_{side}",
            mirror_x(lambda: soft(cyl_x(38, 62, 88, 328, 650), 5).union(cyl_x(20, 87, 89.5, 328, 650)), sx),
            hl("orange", "dark", "B"),
        )
    holder = cyl_z(140, 607, 618).union(box(-35, 35, AX, 330, 607, 618))
    holder = holder.cut(cyl_z(70, 600, 625))
    add("condenser_holder", soft(holder, 3), "white")
    for i, ang in enumerate((-150, -30)):
        s = soft(cyl_x(10, 66, 86, AX, 612.5), 2).rotate((0, AX, 0), (0, AX, 1), ang)
        add(f"C_centering_screw_{i}", s, hl("cobalt", "dark", "C"))
    tc = AX + 12
    add("condenser_turret", soft(cyl_z(185, 575, 607, 0, tc), 4), "white")
    band = cyl_z(188, 584, 598, 0, tc).cut(cyl_z(175, 580, 600, 0, tc))
    add("condenser_turret_band", band, "dark")
    lever = soft(cyl_x(7, 90, 112, tc, 591), 2).rotate((0, tc, 0), (0, tc, 1), -105)
    add("aperture_lever", lever, hl("green", "white", "lever"))
    add("condenser_lens", cyl_z(80, 516, 575), "white")
    add("condenser_lens_rim", soft(cyl_z(82, 508, 516), 2), "dark")
    return P


def _smooth_groups(faces):
    """Group faces that meet tangentially, so no outline is drawn there."""
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges, pairs = {}, []
    for i, f in enumerate(faces):
        for e in f.Edges():
            edges.setdefault(e.hashCode(), []).append((i, e))
    for lst in edges.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                (i, e), (j, e2) = lst[a], lst[b]
                if i == j or not e.isSame(e2):
                    continue
                try:
                    p = e.positionAt(0.5)
                    if faces[i].normalAt(p).dot(faces[j].normalAt(p)) > 0.97:
                        parent[find(i)] = find(j)
                        pairs.append((i, j))
                except Exception:
                    pass
    return [find(i) for i in range(len(faces))], pairs


def _axis_tone(n):
    """Tone index per normal: 0 top, 1 front, 2 side, 3 back or underside.

    Parameters
    ----------
    n : ndarray, shape (m, 3)
        Outward normals.

    Returns
    -------
    ndarray, shape (m,)
    """
    ax = np.abs(n).argmax(axis=1)
    sign = np.take_along_axis(n, ax[:, None], axis=1)[:, 0] > 0
    tone = np.full(len(n), 2)
    tone[(ax == 2) & sign] = 0
    tone[(ax == 2) & ~sign] = 3
    tone[(ax == 1) & ~sign] = 1
    tone[(ax == 1) & sign] = 3
    return tone


def _face_shading(face, verts, tris, force=False):
    """Fixed tone for a face, or per-vertex normals for a blend.

    Returns
    -------
    tone : int
        Tone index, or -1 when the face is a blend (fillet or corner patch)
        that should be toned per pixel from its normals.
    normals : ndarray, shape (len(verts), 3)
        Outward vertex normals (zeros unless the face is a blend).
    """
    v = np.array([(q.x, q.y, q.z) for q in verts])
    t = np.array(tris)
    zeros = np.zeros_like(v)
    if face.geomType() == "PLANE":
        nv = face.normalAt()
        return int(_axis_tone(np.array([[nv.x, nv.y, nv.z]]))[0]), zeros
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    total = np.linalg.norm(n, axis=1).sum()
    mean = n.sum(axis=0)
    wraps = total == 0 or np.linalg.norm(mean) / total < 0.8
    if wraps and not force:
        return 2, zeros
    try:
        k = int(np.linalg.norm(n, axis=1).argmax())
        nv = face.normalAt(cq.Vector(*v[t[k]].mean(axis=0)))
        if np.dot((nv.x, nv.y, nv.z), n[k]) < 0:
            n = -n
    except Exception:
        pass
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, t[:, k], n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
    return -1, vn


def mesh_scene(parts):
    """Tessellate parts into one triangle soup with per-face attributes.

    Returns
    -------
    dict
        ``pts`` (n, 3), ``normals`` (n, 3), ``tris`` (m, 3), ``tri_id`` (m,),
        and per-face-id tables ``fill`` (k, 4, 3), ``line`` (k, 3),
        ``group`` (k,), ``tone`` (k,). Face id 0 is the background.
    """
    pts, nrm, tris, tri_id = [], [], [], []
    white = [(255, 255, 255)] * 4
    fill, line, group, tone = [white], [(255, 255, 255)], [0], [0]
    n, gbase = 0, 1
    for _, wp, pal in parts:
        flat, smooth = pal.endswith("!"), pal.endswith("~")
        tones = PAL[pal.rstrip("!~")]
        shape = wp.val()
        shape.mesh(0.2, 0.06)
        faces = shape.Faces()
        groups, _ = _smooth_groups(faces)
        for f, g in zip(faces, groups):
            verts, t = f.tessellate(0.2, 0.06)
            if not t:
                continue
            ft, vn = _face_shading(f, verts, t, smooth)
            if flat:
                ft = 2
            fid = len(fill)
            fill.append(tones[:4])
            line.append(tones[4])
            group.append(gbase + g)
            tone.append(ft)
            pts.extend((v.x, v.y, v.z) for v in verts)
            nrm.extend(vn)
            tris.extend((a + n, b + n, c + n) for a, b, c in t)
            tri_id.extend([fid] * len(t))
            n += len(verts)
        gbase += len(faces)
    return dict(
        pts=np.array(pts),
        normals=np.array(nrm),
        tris=np.array(tris),
        tri_id=np.array(tri_id),
        fill=np.array(fill, np.uint8),
        line=np.array(line, np.uint8),
        group=np.array(group, np.int64),
        tone=np.array(tone),
    )


def render(scene, path, focus, azim, elev, half_height, size=(960, 1060), ss=3, line_px=5):
    """Render a flat-shaded cartoon with outlines.

    Parameters
    ----------
    scene : dict
        Output of `mesh_scene`.
    path : path-like
        Output PNG.
    focus : tuple of float
        Point at the image centre.
    azim : float
        Camera azimuth in degrees, measured from +x toward the front (-y).
    elev : float
        Camera elevation in degrees.
    half_height : float
        Half the image height in model units (orthographic).
    size : tuple of int
        Output size in pixels.
    ss : int
        Supersampling factor.
    line_px : int
        Outline width in supersampled pixels (odd).
    """
    tris, tri_id = scene["tris"], scene["tri_id"]
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(ns.numpy_to_vtk(scene["pts"], deep=True))
    pd.SetPoints(vp)
    cells = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64)
    ca = vtk.vtkCellArray()
    ca.SetCells(len(tris), ns.numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
    pd.SetPolys(ca)
    idc = np.stack([tri_id & 255, (tri_id >> 8) & 255, (tri_id >> 16) & 255], axis=1).astype(np.uint8)
    ncol = np.clip(np.rint((scene["normals"] + 1) * 127.5), 0, 255)
    carr = ns.numpy_to_vtk(idc, deep=True)
    carr.SetName("id")
    parr = ns.numpy_to_vtk(ncol.astype(np.uint8), deep=True)
    parr.SetName("normal")
    pd.GetCellData().SetScalars(carr)
    pd.GetPointData().SetScalars(parr)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    ren.AddActor(actor)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(0)
    win.SetSize(size[0] * ss, size[1] * ss)
    win.AddRenderer(ren)
    a, e = math.radians(azim), math.radians(elev)
    d = np.array([math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), math.sin(e)])
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(*focus)
    cam.SetPosition(*(np.array(focus) + 4000 * d))
    cam.SetViewUp(0, 0, 1)
    cam.SetParallelScale(half_height)
    ren.ResetCameraClippingRange()

    def grab():
        win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img = w2i.GetOutput()
        w, h, _ = img.GetDimensions()
        buf = ns.vtk_to_numpy(img.GetPointData().GetScalars())
        return buf.reshape(h, w, -1)[::-1, :, :3].astype(np.int64)

    mapper.SetScalarModeToUseCellData()
    rgb = grab()
    mapper.SetScalarModeToUsePointData()
    normal = grab() / 127.5 - 1.0
    ids = rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)
    ids = np.clip(ids, 0, len(scene["fill"]) - 1)
    tone = scene["tone"][ids]
    blend = tone < 0
    tone[blend] = 2
    color = scene["fill"][ids, tone]
    # blends fade between the tones of the faces they join
    nb = normal[blend]
    nb /= np.maximum(np.linalg.norm(nb, axis=1), 1e-9)[:, None]
    fb = scene["fill"][ids[blend]].astype(float)
    rows = np.arange(len(nb))
    mix = (
        nb[:, 0:1] ** 2 * fb[:, 2]
        + nb[:, 1:2] ** 2 * fb[rows, np.where(nb[:, 1] < 0, 1, 3)]
        + nb[:, 2:3] ** 2 * fb[rows, np.where(nb[:, 2] > 0, 0, 3)]
    )
    color[blend] = np.clip(np.rint(mix), 0, 255).astype(np.uint8)
    kid = scene["group"][ids]
    oc = scene["line"][ids]
    lines = np.full_like(color, 255)
    eh = kid[:, :-1] != kid[:, 1:]
    m = np.minimum(oc[:, :-1], oc[:, 1:])
    lines[:, :-1][eh] = np.minimum(lines[:, :-1][eh], m[eh])
    ev = kid[:-1, :] != kid[1:, :]
    m = np.minimum(oc[:-1, :], oc[1:, :])
    lines[:-1, :][ev] = np.minimum(lines[:-1, :][ev], m[ev])
    lim = Image.fromarray(lines).filter(ImageFilter.MinFilter(line_px))
    la = np.asarray(lim)
    mask = (la < 255).any(axis=2)
    out = color.copy()
    out[mask] = la[mask]
    Image.fromarray(out).resize(size, Image.LANCZOS).save(path)


VIEWS = [
    # name, title, focus, azim, elev, half_height, highlight
    ("ref_match", "Whole stand (reference angle)", (-25, 249, 402), 35, 10, 540, False),
    ("ref_match_hl", "Whole stand, Koehler controls coloured", (-25, 249, 402), 35, 10, 540, True),
    ("illum_right", "Illuminator from front-right", (0, AX, 640), 62, 10, 170, "close"),
    ("illum_front", "Illuminator from front", (0, AX, 640), 90, 6, 170, "close"),
    ("stand_left", "Whole stand from front-left", (0, 285, 440), 140, 12, 520, False),
    ("stand_front", "Whole stand from front", (0, 285, 440), 90, 8, 520, False),
]


def main(out, ref=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plain = build()
    hi = build(True)
    close = [q for q in hi if not q[0].startswith(("eyepiece", "tube_"))]
    scenes = {False: mesh_scene(plain), True: mesh_scene(hi), "close": mesh_scene(close)}
    for name, _, focus, az, el, hh, h in VIEWS:
        render(scenes[h], out / f"{name}.png", focus, az, el, hh)
    assy = cq.Assembly(name="cartoon_inverted_microscope")
    for name, wp, pal in plain:
        r, g, b = PAL[pal.rstrip("!~")][2]
        assy.add(wp, name=name, color=cq.Color(r / 255, g / 255, b / 255))
    assy.export(str(out / "cartoon_scope.step"))
    assy.export(str(out / "cartoon_scope.glb"))
    with PdfPages(out / "summary.pdf") as pdf:
        pages = []
        if ref and Path(ref).exists():
            pages.append([("Reference", ref), ("Model render", out / "ref_match.png")])
        rest = [(t, out / f"{n}.png") for n, t, *_ in VIEWS[1:]]
        pages += [rest[i : i + 2] for i in range(0, len(rest), 2)]
        for group in pages:
            fig, axes = plt.subplots(1, len(group), figsize=(11, 6.5), squeeze=False)
            for ax, (title, p) in zip(axes[0], group):
                ax.imshow(plt.imread(p))
                ax.set_title(title, fontsize=11)
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, dpi=170)
            plt.close(fig)
    print("parts:", len(plain))


if __name__ == "__main__":
    main(*sys.argv[1:])
