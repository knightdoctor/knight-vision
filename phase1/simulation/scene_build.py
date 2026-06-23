"""Programmatic Knight Vision MVP scene — bpy script.

Builds a parameterised scene with:
- Cot frame (visual reference, not load-bearing)
- Mattress plane
- Infant placeholder: capsule torso + sphere head (Week 1 — primitives;
  upgrade to rigged commercial mesh in Week 2/3)
- Chest expansion shape key on the torso (drives breathing animation)
- Two cameras at MVP L-overhang positions (Cam A top of arm, Cam B on
  pole 200 mm below)
- 850 nm IR illuminator (point light, simulated as visible-spectrum
  white light for rendering purposes — IR-band filtering happens in
  depth.py colour-channel selection)

Invocation:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python phase1/simulation/scene_build.py -- \
        --out phase1/simulation/scene.blend

Coordinate convention inside Blender:
    +X right, +Y forward (away from sensor), +Z up.
    Subject lies on the mattress at the origin, head pointing +Y.
    Cameras above and behind (low-Y), pointing toward +Y and -Z.

depth.py converts to the shared Knight Vision frame
(X right, Y up, Z forward) when projecting to point clouds.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy

# ── MVP geometry defaults (matches mvp_sensor_stack_architecture spec) ───────
DEFAULTS = {
    # All units metres / radians unless noted.
    "cot_inner_size":      (0.60, 1.20, 0.60),  # W, L, H
    "cot_wall_thickness":  0.02,
    "mattress_size":       (0.50, 1.00, 0.08),
    "mattress_top_z":      0.10,                # mattress top above cot base

    "infant_torso_length": 0.30,                # 6-month placeholder
    "infant_torso_radius": 0.08,                # capsule radius (~8 cm)
    "infant_head_radius":  0.07,
    "infant_y_offset":     0.10,                # torso centre Y (head side +Y)
    "chest_excursion_mm":  5.0,                 # default for scene; render.py overrides

    # Cameras — Week-2 #2 B1 parallel-axis L-overhang.
    # See feedback in dev_log_2026-06-22.md: strict world-vertical baseline
    # + pitched cams gives Tz residual that forces R1 to rotate chest off-
    # image; offsetting Cam B along cam-local −Y keeps R1 ≈ I. 100 mm
    # baseline chosen so chest disp ≈ 222 px fits inside image_width/2
    # = 400 px at 0.50 m working distance.
    "chest_centre_z":      None,                # computed = mattress_top_z + infant_torso_radius
    "cam_a_offset":        (0.00, -0.433, 0.25),
    "cam_b_offset":        (0.00, -0.483, 0.163),
    "cam_parallel_pitch_deg": 30.0,
    "cam_a_pitch_deg":     0.0,                 # unused in parallel-axis mode
    "cam_b_pitch_deg":     0.0,
    "cam_resolution":      (1280, 800),
    "cam_horiz_fov_deg":   60.0,

    # Lighting
    "ir_led_offset":       (0.00, -0.10, 0.55), # roughly co-located with Cam A
    "ir_led_energy":       50.0,                # watts (passive scene power)

    # Texture / projector variants for B1 surface-texture comparison
    # (control / cloth / projector). Set via --variant CLI arg in main().
    "variant":             "control",
    # Cloth variant — noise on the torso material's albedo + bump
    "cloth_noise_scale":   100.0,   # cycles per noise-unit; coarser weave that survives sub-pixel sampling
    "cloth_albedo_contrast": 0.30,  # ± fraction around base albedo (moderate; not skin-physical, but realistic for textured swaddle/blanket)
    "cloth_bump_strength": 0.10,    # micro-relief
    # Projector variant — overhead emission plane with a sparse-dot
    # noise texture (Intel RealSense-style speckle for stereo correspondence)
    "projector_height_above_chest": 0.40,    # m, plane height above chest
    "projector_spot_size_deg":      50.0,    # unused with emission-plane impl
    "projector_energy":             6.0,     # W/m² emission peak per dot
    "projector_noise_scale":        50.0,    # generated-coord scale; sparse projection dots
    "projector_dot_density_thresh": 0.70,    # color-ramp threshold; higher = sparser dots
}


def clear_scene():
    """Wipe all objects, lights, cameras."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def make_material(name: str, base_color: tuple, roughness: float = 0.5) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_fabric_material(name: str, base_color: tuple,
                         noise_scale: float, albedo_contrast: float,
                         bump_strength: float,
                         roughness: float = 0.7) -> bpy.types.Material:
    """Procedural fabric: noise texture modulates albedo by ±contrast and
    drives a subtle bump. Spatial frequency = noise_scale cycles per metre
    (≈ 1/scale metres per cycle). Used for the B1 'cloth' variant — gives
    SGBM matchable surface texture that's still realistic for fabric.

    Uses Generated texture coords (object-bound, deterministic across the
    surface) rather than the default Spherical UV which collapses noise
    near the UV poles.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    # Generated coords → noise → albedo modulation
    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.5
    links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])

    # Push noise through a color ramp to sharpen contrast: low end darker,
    # high end at base.
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = 0.3
    dark = max(0.0, 1.0 - 2.0 * albedo_contrast)
    ramp.color_ramp.elements[0].color = (
        base_color[0] * dark, base_color[1] * dark, base_color[2] * dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (*base_color, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    if bump_strength > 0:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_strength
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_ir_projector(params):
    """Variant 'projector': overhead emission plane with a noise-driven
    sparse dot pattern in its texture. Approximates a RealSense-style
    active-stereo IR speckle projector.

    Implementation note: Eevee's shader-node support on light data is
    flaky for spot-light textures — the noise → emission pattern on a
    LIGHT object often renders as a uniform colour wash. Instead, we use
    a downward-facing emission plane held above the cot, with the dot
    pattern baked into the plane's material. The plane casts the speckle
    onto everything below it via direct illumination."""
    chest_z = params["mattress_top_z"] + params["infant_torso_radius"]
    yoff = params["infant_y_offset"]
    height = params["projector_height_above_chest"]

    plane_loc = (0.0, yoff, chest_z + height)
    bpy.ops.mesh.primitive_plane_add(size=0.40, location=plane_loc)
    plane = bpy.context.active_object
    plane.name = "ir_projector_plane"
    # Flip so the emission face points DOWN (-Z world)
    plane.rotation_euler = (math.radians(180.0), 0.0, 0.0)

    # Material: noise → color ramp (sparse dots) → emission
    mat = bpy.data.materials.new(name="ir_projector_pattern")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = params["projector_noise_scale"]
    noise.inputs["Detail"].default_value = 0.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = params["projector_dot_density_thresh"]
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    bright = float(params["projector_energy"])
    ramp.color_ramp.elements[1].color = (bright, bright, bright, 1.0)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(mat)
    return plane


def add_cot(params):
    """Build cot as a rectangular frame: 4 wall planes + bottom."""
    w, l, h = params["cot_inner_size"]
    t = params["cot_wall_thickness"]
    mat = make_material("cot_frame", (0.30, 0.22, 0.16), roughness=0.7)

    def panel(name, location, size):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        ob = bpy.context.active_object
        ob.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
        ob.name = name
        ob.data.materials.append(mat)
        return ob

    panel("cot_floor",  (0,        0,           0),         (w + 2 * t, l + 2 * t, t))
    panel("cot_wall_W", (-(w / 2 + t / 2), 0,    h / 2),    (t,         l + 2 * t, h))
    panel("cot_wall_E", ( (w / 2 + t / 2), 0,    h / 2),    (t,         l + 2 * t, h))
    panel("cot_wall_S", (0,       -(l / 2 + t / 2), h / 2), (w,         t,         h))
    panel("cot_wall_N", (0,        (l / 2 + t / 2), h / 2), (w,         t,         h))


def add_mattress(params):
    mw, ml, mh = params["mattress_size"]
    z = params["mattress_top_z"] - mh / 2
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, z))
    ob = bpy.context.active_object
    ob.scale = (mw / 2, ml / 2, mh / 2)
    ob.name = "mattress"
    mat = make_material("mattress", (0.85, 0.85, 0.82), roughness=0.85)
    ob.data.materials.append(mat)


def add_infant(params):
    """Capsule torso + sphere head. Torso has a 'chest_expansion' shape key
    that scales it slightly along +Z; render.py keyframes the value."""
    tl = params["infant_torso_length"]
    tr = params["infant_torso_radius"]
    hr = params["infant_head_radius"]
    yoff = params["infant_y_offset"]
    chest_z = params["mattress_top_z"] + tr        # chest centre above mattress

    # Torso — a stretched UV sphere (cheap capsule). Long axis is Y.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0, location=(0.0, yoff, chest_z), segments=48, ring_count=24)
    torso = bpy.context.active_object
    torso.name = "infant_torso"
    torso.scale = (tr, tl / 2, tr)
    if params.get("variant") == "cloth":
        skin = make_fabric_material(
            "torso_fabric", (0.85, 0.70, 0.62),
            noise_scale=params["cloth_noise_scale"],
            albedo_contrast=params["cloth_albedo_contrast"],
            bump_strength=params["cloth_bump_strength"])
    else:
        skin = make_material("skin", (0.85, 0.70, 0.62), roughness=0.55)
    torso.data.materials.append(skin)
    # Apply scale so the rest pose has scale=1 → shape keys + lateral
    # rendering match expectations.
    bpy.context.view_layer.objects.active = torso
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Add a shape key that lifts the upper-front (toward camera, +Z) by
    # `chest_excursion_mm` mm. Indexing: select vertices with Z > chest_z
    # and bias upward.
    torso.shape_key_add(name="Basis")
    sk = torso.shape_key_add(name="chest_expansion")
    sk.value = 0.0  # render.py drives this
    excursion = params["chest_excursion_mm"] / 1000.0
    for i, v in enumerate(torso.data.vertices):
        # Push only the front upper half (+Z, +X facing camera side) outward.
        # In the rest pose, +Z is up (Cam A and Cam B view from -Y so they
        # see the chest from below as well). Treat +Z half as "front".
        if v.co.z > 0:
            # Gaussian-weighted bump centred at the apex
            w_z = max(0.0, v.co.z / tr)
            w_y = math.exp(-((v.co.y - 0) / (tl * 0.25)) ** 2)
            sk.data[i].co = (
                v.co.x,
                v.co.y,
                v.co.z + excursion * w_z * w_y,
            )

    # Head — simple sphere, no shape key.
    head_y = yoff + tl / 2 + hr * 0.85
    head_z = chest_z + hr * 0.3
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=hr, location=(0.0, head_y, head_z), segments=32, ring_count=16)
    head = bpy.context.active_object
    head.name = "infant_head"
    head.data.materials.append(skin)


def add_camera(name: str, location, pitch_deg: float, params) -> bpy.types.Object:
    """Place a camera at `location`. Parallel-axis mode (if
    cam_parallel_pitch_deg set) gives every cam the same orientation;
    otherwise falls back to look-at-target."""
    cam_data = bpy.data.cameras.new(name=name + "_data")
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    parallel = params.get("cam_parallel_pitch_deg")
    if parallel is not None:
        cam.rotation_euler = (math.radians(90.0 - parallel), 0.0, 0.0)
    else:
        target = (0.0, params["infant_y_offset"], params["mattress_top_z"]
                  + params["infant_torso_radius"])
        cam.rotation_euler = _look_at_euler(cam.location, target,
                                            pitch_offset_deg=pitch_deg)
    # Intrinsics
    res_x, res_y = params["cam_resolution"]
    cam_data.sensor_width = 6.4  # mm — FLIR Blackfly S sensor approximate
    cam_data.lens = (cam_data.sensor_width / 2.0) / math.tan(
        math.radians(params["cam_horiz_fov_deg"]) / 2.0)
    return cam


def _look_at_euler(loc, target, pitch_offset_deg: float = 0.0):
    """Return Euler angles so a default-orientation Blender camera (looking
    along -Z) instead looks from `loc` toward `target`, with an optional
    extra down-pitch."""
    import mathutils
    direction = mathutils.Vector(target) - mathutils.Vector(loc)
    rot_quat = direction.to_track_quat("-Z", "Y")
    eul = rot_quat.to_euler()
    # Apply down-pitch around the camera's local +X (which is "right" after
    # the track-to). Positive pitch_offset_deg tilts the lens further down.
    if pitch_offset_deg != 0:
        from math import radians
        rot_pitch = mathutils.Matrix.Rotation(
            radians(pitch_offset_deg), 4, "X")
        eul = (rot_quat.to_matrix().to_4x4() @ rot_pitch).to_euler()
    return eul


def add_lighting(params):
    """One 850 nm IR illuminator near Cam A + a dim fill so renders aren't
    pitch-black. Both rendered as white visible-spectrum light; depth.py
    pulls the green channel as a stand-in for monochrome IR sensitivity."""
    chest_z = params["mattress_top_z"] + params["infant_torso_radius"]
    loc = (params["ir_led_offset"][0],
           params["ir_led_offset"][1],
           chest_z + params["ir_led_offset"][2])
    bpy.ops.object.light_add(type="POINT", location=loc)
    led = bpy.context.active_object
    led.name = "ir_led"
    led.data.energy = params["ir_led_energy"]
    led.data.color = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="SUN", location=(2.0, -2.0, 3.0))
    fill = bpy.context.active_object
    fill.name = "ambient_fill"
    fill.data.energy = 0.3
    fill.data.color = (0.8, 0.85, 1.0)


def configure_render(params):
    """Render settings — Eevee for speed, grayscale-friendly output."""
    scene = bpy.context.scene
    # Blender 5.x uses "BLENDER_EEVEE_NEXT" or "BLENDER_EEVEE"; fall back.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "EEVEE"):
        try:
            scene.render.engine = engine
            break
        except (TypeError, AttributeError):
            continue
    scene.render.resolution_x, scene.render.resolution_y = params["cam_resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"  # 8-bit greyscale
    scene.render.fps = 30
    scene.render.film_transparent = False
    # Eevee defaults are fine for a smoke test — no AO, no bloom, no SSR.


def main():
    argv = sys.argv
    # Blender passes everything after "--" as user args
    if "--" in argv:
        user_argv = argv[argv.index("--") + 1:]
    else:
        user_argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write scene.blend")
    ap.add_argument("--chest-excursion-mm", type=float, default=None,
                    help="Override chest excursion (used by shape key)")
    ap.add_argument("--variant", type=str, default=None,
                    choices=["control", "cloth", "projector"],
                    help="B1 surface-texture variant")
    args = ap.parse_args(user_argv)

    params = dict(DEFAULTS)
    if args.chest_excursion_mm is not None:
        params["chest_excursion_mm"] = args.chest_excursion_mm
    if args.variant is not None:
        params["variant"] = args.variant

    clear_scene()
    add_cot(params)
    add_mattress(params)
    add_infant(params)

    chest_z = params["mattress_top_z"] + params["infant_torso_radius"]
    yoff = params["infant_y_offset"]
    cam_a_loc = (params["cam_a_offset"][0],
                 yoff + params["cam_a_offset"][1],
                 chest_z + params["cam_a_offset"][2])
    cam_b_loc = (params["cam_b_offset"][0],
                 yoff + params["cam_b_offset"][1],
                 chest_z + params["cam_b_offset"][2])
    add_camera("Cam_A", cam_a_loc, params["cam_a_pitch_deg"], params)
    add_camera("Cam_B", cam_b_loc, params["cam_b_pitch_deg"], params)
    add_lighting(params)
    if params.get("variant") == "projector":
        add_ir_projector(params)
    configure_render(params)

    # Save intrinsics + extrinsics alongside scene.blend so depth.py doesn't
    # have to re-parse the .blend file.
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsics = {
        "cam_a": {
            "location": cam_a_loc,
            "pitch_deg": params["cam_a_pitch_deg"],
        },
        "cam_b": {
            "location": cam_b_loc,
            "pitch_deg": params["cam_b_pitch_deg"],
        },
        # Parallel-axis pitch tells depth.py / smoke_test.py to construct
        # camera world rotation analytically from this shared value rather
        # than computing look-at-chest from each cam's position.
        "parallel_pitch_deg": params.get("cam_parallel_pitch_deg"),
        "baseline_m":      cam_a_loc[2] - cam_b_loc[2],   # vertical baseline
        "resolution":      params["cam_resolution"],
        "horiz_fov_deg":   params["cam_horiz_fov_deg"],
        "chest_centre_world": (0.0, yoff, chest_z),
        "chest_excursion_mm": params["chest_excursion_mm"],
        "mattress_top_z":  params["mattress_top_z"],
        "variant":         params.get("variant", "control"),
        "shared_frame_note": (
            "Blender frame is X right, Y forward (away from sensor), Z up. "
            "depth.py converts to shared Knight Vision frame "
            "(X right, Y up, Z forward) via permutation (x, z, y)."),
    }
    (out_path.parent / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"[scene_build] wrote {out_path}")
    print(f"[scene_build] wrote {out_path.parent / 'intrinsics.json'}")


if __name__ == "__main__":
    main()
