# ==============================================================================
# Yutao Liu s4213211
# Modern Game AI Algorithms Assignments: A1 Minecraft
#
# Reference: GPDC Documentation website: Tutorial - building a house
# https://gdpc.readthedocs.io/en/stable/.
#Important : enter /setbuildarea ~0 0 ~0 ~128 255 ~128 in the Minecraft Chatbox(Press T) first!
#
# ==============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice, random
from gdpc import Editor, Block
from gdpc.geometry import placeCuboid, placeCuboidHollow
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------
# Global variables: Store the generated structures as an array for overlap detection
# ---------------------------
existingStructures = []  # Each element: (x, z, width, depth, type) - Comprehensive list of all structures

# ---------------------------
# Building style functions: Define different architectural styles and related functions
# ---------------------------

# Define different building styles
BUILDING_STYLES = {
    "forest": {
        "walls": [
            Block("oak_planks"),
            Block("spruce_planks"),
            Block("dark_oak_planks")
        ],
        "accent": [
            Block("spruce_log"),
            Block("oak_log")
        ],
        "roof": [
            "spruce_stairs",
            "oak_stairs",
            "dark_oak_stairs"
        ],
        "roof_material": [
            "spruce_planks",
            "oak_planks",
            "dark_oak_planks"
        ],
        "floor": [
            Block("oak_planks"),
            Block("spruce_planks")
        ],
        "decorations": [
            Block("lantern"),
            Block("potted_fern"),
            Block("bookshelf"),
            Block("barrel")
        ]
    },
    "desert": {
        "walls": [
            Block("smooth_sandstone"),
            Block("cut_sandstone"),
            Block("sandstone")
        ],
        "accent": [
            Block("chiseled_sandstone"),
            Block("orange_terracotta")
        ],
        "roof": [
            "smooth_sandstone_stairs",
            "sandstone_stairs"
        ],
        "roof_material": [
            "smooth_sandstone",
            "cut_sandstone"
        ],
        "floor": [
            Block("smooth_sandstone"),
            Block("terracotta")
        ],
        "decorations": [
            Block("potted_cactus"),
            Block("potted_dead_bush"),
            Block("flower_pot"),
            Block("hay_block")
        ]
    },
    "stone": {
        "walls": [
            Block("stone_bricks"),
            Block("cobblestone"),
            Block("andesite")
        ],
        "accent": [
            Block("polished_andesite"),
            Block("mossy_stone_bricks")
        ],
        "roof": [
            "stone_brick_stairs",
            "cobblestone_stairs",
            "andesite_stairs"
        ],
        "roof_material": [
            "stone_bricks",
            "cobblestone",
            "andesite"
        ],
        "floor": [
            Block("polished_andesite"),
            Block("stone_bricks")
        ],
        "decorations": [
            Block("lantern"),
            Block("campfire"),
            Block("brewing_stand"),
            Block("cauldron")
        ]
    }
}


def s4213211_selectBuildingStyle(editor, x, z, heightmap, x_min, z_min):
    """Selects appropriate building style based on ground block type, including desert, stone and forest"""
    local_x = x - x_min
    local_z = z - z_min
    y = int(heightmap[local_x, local_z]) - 1

    ground_block = editor.getBlock((x, y, z))

    if "sand" in str(ground_block) or "sandstone" in str(ground_block):
        return "desert"
    elif "stone" in str(ground_block) or "andesite" in str(ground_block) or "granite" in str(ground_block):
        return "stone"
    else:
        return "forest"  # Default forest style


def s4213211_getBuildingMaterials(building_style):
    """Gets building materials for a specific style"""
    style_data = BUILDING_STYLES[building_style]
    materials = {
        "wall": choice(style_data["walls"]),
        "accent": choice(style_data["accent"]),
        "roof_stair": choice(style_data["roof"]),
        "roof_plank": choice(style_data["roof_material"]),
        "floor": choice(style_data["floor"]),
        "decorations": style_data["decorations"]
    }
    return materials


# ---------------------------
# Detect functions: Here are the functions for detecting overlap and terrain analysis
# ---------------------------

def s4213211_isOverlap(new_x, new_z, new_width, new_depth, boxes, min_distance=2):
    """Checks for overlap between structures using AABB collision detection"""
    padded_x = new_x - min_distance
    padded_z = new_z - min_distance
    padded_width = new_width + 2 * min_distance
    padded_depth = new_depth + 2 * min_distance

    new_x2 = padded_x + padded_width
    new_z2 = padded_z + padded_depth

    for box in boxes:
        if len(box) >= 4:
            ex, ez = box[0], box[1]
            ew, ed = box[2], box[3]
            ex2 = ex + ew
            ez2 = ez + ed
            overlaps_on_x_axis = new_x2 > ex and padded_x < ex2
            overlaps_on_z_axis = new_z2 > ez and padded_z < ez2
            if overlaps_on_x_axis and overlaps_on_z_axis:
                return True
    return False


def s4213211_adaptiveFoundation(editor, x, z, width, depth, heightmap, x_min, z_min):
    """Creates a foundation that adapts to the terrain height"""
    heights = []
    for dx in range(width):
        for dz in range(depth):
            local_x = x + dx - x_min
            local_z = z + dz - z_min

            local_x = max(0, min(local_x, len(heightmap) - 1))
            local_z = max(0, min(local_z, len(heightmap[0]) - 1))

            heights.append(int(heightmap[local_x, local_z]))

    target_y = sum(heights) // len(heights)

    water_present = False
    for dx in range(width):
        for dz in range(depth):
            for y_check in range(target_y - 3, target_y + 1):
                if editor.getBlock((x + dx, y_check, z + dz)) == Block("water"):
                    water_present = True
                    target_y = max(target_y, y_check + 2)

    for dx in range(width):
        for dz in range(depth):
            local_x = x + dx - x_min
            local_z = z + dz - z_min

            local_x = max(0, min(local_x, len(heightmap) - 1))
            local_z = max(0, min(local_z, len(heightmap[0]) - 1))

            current_y = int(heightmap[local_x, local_z])

            if current_y < target_y:
                for y in range(current_y, target_y):
                    editor.placeBlock((x + dx, y, z + dz), Block("stone"))

            elif current_y > target_y:
                for y in range(target_y, current_y):
                    editor.placeBlock((x + dx, y, z + dz), Block("air"))

    if water_present:
        pillar_spacing = 3
        for dx in range(0, width, pillar_spacing):
            if dx + 1 >= width and dx != 0:
                dx = width - 1

            for dz in range(0, depth, pillar_spacing):
                if dz + 1 >= depth and dz != 0:
                    dz = depth - 1

                local_x = x + dx - x_min
                local_z = z + dz - z_min

                local_x = max(0, min(local_x, len(heightmap) - 1))
                local_z = max(0, min(local_z, len(heightmap[0]) - 1))

                ground_y = int(heightmap[local_x, local_z])

                for y in range(ground_y, target_y):
                    editor.placeBlock((x + dx, y, z + dz), Block("oak_log"))

    return target_y


# ---------------------------
# Generated House functions: all functions related to building a house in Minecraft, including generating the main structure, roof, doors, windows and interior decorations.
# ---------------------------
def s4213211_generateHouse(editor, x, y, z, width, height, depth, wallBlock, floorPalette):
    """Creates the main house structure with walls and floor"""
    placeCuboidHollow(editor, (x, y, z), (x + width, y + height, z + depth), wallBlock)
    placeCuboid(editor, (x, y, z), (x + width, y - 5, z + depth), floorPalette)
    placeCuboid(editor, (x + 1, y + 1, z + 1), (x + width - 1, y + height - 1, z + depth - 1), Block("air"))


def s4213211_generateRoof(editor, x, y, z, width, height, depth, roof_stair_type, roof_plank_type):
    """Creates a roof with either triangular or flat style which probability is 50% each"""
    roof_type = "triangular" if random() < 0.5 else "flat"

    if roof_type == "triangular":
        steps_needed = (width + 2) // 2 + 1
        max_height = min(5, steps_needed)

        base_y = y + height
        placeCuboid(editor, (x - 1, base_y, z - 1), (x + width + 1, base_y, z + depth + 1), Block(roof_plank_type))

        for step in range(1, max_height + 1):
            current_y = base_y + step
            left_edge = x - 1 + step
            right_edge = x + width + 1 - step

            if left_edge > right_edge:
                break

            placeCuboid(editor, (left_edge, current_y, z - 1), (right_edge, current_y, z + depth + 1),
                        Block(roof_plank_type))

            if step < max_height:
                leftStair = Block(roof_stair_type, {"facing": "east"})
                placeCuboid(editor, (left_edge - 1, current_y, z - 1), (left_edge - 1, current_y, z + depth + 1),
                            leftStair)

                rightStair = Block(roof_stair_type, {"facing": "west"})
                placeCuboid(editor, (right_edge + 1, current_y, z - 1), (right_edge + 1, current_y, z + depth + 1),
                            rightStair)

    else:  # Flat roof
        yy = y + height

        placeCuboid(editor, (x - 2, yy, z - 2), (x + width + 2, yy, z + depth + 2), Block(roof_plank_type))

        northBlock = Block(roof_stair_type, {"facing": "south"})
        placeCuboid(editor, (x - 2, yy, z - 2), (x + width + 2, yy, z - 2), northBlock)

        southBlock = Block(roof_stair_type, {"facing": "north"})
        placeCuboid(editor, (x - 2, yy, z + depth + 2), (x + width + 2, yy, z + depth + 2), southBlock)

        eastBlock = Block(roof_stair_type, {"facing": "west"})
        placeCuboid(editor, (x + width + 2, yy, z - 1), (x + width + 2, yy, z + depth + 1), eastBlock)

        westBlock = Block(roof_stair_type, {"facing": "east"})
        placeCuboid(editor, (x - 2, yy, z - 1), (x - 2, yy, z + depth + 1), westBlock)


def s4213211_generateStyledWindows(editor, x, y, z, width, height, depth, building_style):
    """Creates windows on all four walls and based on building style"""
    win_width = 2
    win_height = 2

    if building_style == "forest":
        window_chance = 70
    elif building_style == "desert":
        window_chance = 50
    else:  # stone
        window_chance = 40

    walls = [
        {"name": "north", "position": z, "is_x_fixed": False, "fixed_coord": z},
        {"name": "south", "position": z + depth - 1, "is_x_fixed": False, "fixed_coord": z + depth - 1},
        {"name": "east", "position": x + width - 1, "is_x_fixed": True, "fixed_coord": x + width - 1},
        {"name": "west", "position": x, "is_x_fixed": True, "fixed_coord": x}
    ]

    for wall in walls:
        if randint(0, 100) < window_chance:
            if not wall["is_x_fixed"]:  # North or South wall (x varies, z fixed)
                win_x = randint(x + 1, x + width - win_width - 1)
                win_y = randint(y + 1, y + height - win_height - 1)
                win_z = wall["fixed_coord"]

                for dx in range(win_width):
                    for dy in range(win_height):
                        if building_style == "desert":
                            glass_colors = ["yellow", "orange", "red", "light_blue"]
                            glass_block = Block(f"{choice(glass_colors)}_stained_glass")
                            editor.placeBlock((win_x + dx, win_y + dy, win_z), glass_block)
                        elif building_style == "stone":
                            if randint(0, 100) < 70:
                                editor.placeBlock((win_x + dx, win_y + dy, win_z), Block("iron_bars"))
                            else:
                                editor.placeBlock((win_x + dx, win_y + dy, win_z), Block("glass"))
                        else:  # forest
                            editor.placeBlock((win_x + dx, win_y + dy, win_z), Block("glass"))

                if building_style == "forest":
                    for dy in range(win_height):
                        editor.placeBlock((win_x - 1, win_y + dy, win_z), Block("oak_fence"))
                        editor.placeBlock((win_x + win_width, win_y + dy, win_z), Block("oak_fence"))
                elif building_style == "desert":
                    editor.placeBlock((win_x - 1, win_y, win_z), Block("chiseled_sandstone"))
                    editor.placeBlock((win_x + win_width, win_y, win_z), Block("chiseled_sandstone"))
                    for dx in range(win_width):
                        editor.placeBlock((win_x + dx, win_y - 1, win_z),
                                          Block("smooth_sandstone_slab", {"type": "top"}))
                elif building_style == "stone":
                    facing = "north" if wall["name"] == "south" else "south"
                    opposite = "south" if wall["name"] == "south" else "north"
                    for dx in range(-1, win_width + 1):
                        editor.placeBlock((win_x + dx, win_y - 1, win_z),
                                          Block("stone_brick_stairs", {"facing": facing, "half": "top"}))
                        editor.placeBlock((win_x + dx, win_y + win_height, win_z),
                                          Block("stone_brick_stairs", {"facing": opposite, "half": "top"}))

            else:  # East or West wall (z varies, x fixed)
                win_z = randint(z + 1, z + depth - win_width - 1)
                win_y = randint(y + 1, y + height - win_height - 1)
                win_x = wall["fixed_coord"]

                for dz in range(win_width):
                    for dy in range(win_height):
                        if building_style == "desert":
                            glass_colors = ["yellow", "orange", "red", "light_blue"]
                            glass_block = Block(f"{choice(glass_colors)}_stained_glass")
                            editor.placeBlock((win_x, win_y + dy, win_z + dz), glass_block)
                        elif building_style == "stone":
                            if randint(0, 100) < 70:
                                editor.placeBlock((win_x, win_y + dy, win_z + dz), Block("iron_bars"))
                            else:
                                editor.placeBlock((win_x, win_y + dy, win_z + dz), Block("glass"))
                        else:  # forest
                            editor.placeBlock((win_x, win_y + dy, win_z + dz), Block("glass"))

                if building_style == "forest":
                    for dy in range(win_height):
                        editor.placeBlock((win_x, win_y + dy, win_z - 1), Block("oak_fence"))
                        editor.placeBlock((win_x, win_y + dy, win_z + win_width), Block("oak_fence"))

def s4213211_generateDoor(editor, x, y, z, width):
    """Creates a door on the north side of the house"""
    door_x = x + width // 2

    editor.placeBlock((door_x, y, z), Block("oak_planks"))

    doorBlockBottom = Block("oak_door", {"facing": "north", "half": "lower", "hinge": "left"})
    editor.placeBlock((door_x, y + 1, z), doorBlockBottom)

    doorBlockTop = Block("oak_door", {"facing": "north", "half": "upper", "hinge": "left"})
    editor.placeBlock((door_x, y + 2, z), doorBlockTop)

    placeCuboid(editor, (door_x - 1, y + 1, z - 1),
                (door_x + 1, y + 3, z - 1), Block("air"))


def s4213211_decorateInterior(editor, x, y, z, width, height, depth, building_style, materials):
    """Adds interior decorations based on building style"""
    bed_floor_x = x + 1
    bed_floor_z = z + depth - 2

    editor.placeBlock((bed_floor_x, y, bed_floor_z), materials["floor"])
    editor.placeBlock((bed_floor_x + 1, y, bed_floor_z), materials["floor"])

    bed_foot = Block("red_bed", {"facing": "east", "part": "foot"})
    bed_head = Block("red_bed", {"facing": "east", "part": "head"})
    editor.placeBlock((bed_floor_x, y + 1, bed_floor_z), bed_foot)
    editor.placeBlock((bed_floor_x + 1, y + 1, bed_floor_z), bed_head)

    if building_style == "forest":
        if width > 5:
            editor.placeBlock((x + width - 2, y + 1, z + 2), Block("bookshelf"))
            editor.placeBlock((x + width - 2, y + 2, z + 2), Block("bookshelf"))
        editor.placeBlock((x + width - 2, y + 1, z + depth - 2), Block("crafting_table"))

    elif building_style == "desert":
        editor.placeBlock((x + 2, y + 1, z + 2), Block("potted_dead_bush"))
        editor.placeBlock((x + width - 2, y + 1, z + 2), Block("hay_block"))

    else:  # stone style
        editor.placeBlock((x + width - 2, y + 1, z + 2), Block("brewing_stand"))
        editor.placeBlock((x + 2, y + 1, z + 2), Block("mossy_stone_bricks"))

    s4213211_placeLighting(editor, x, y, z, width, height, depth, building_style)


def s4213211_placeLighting(editor, x, y, z, width, height, depth, building_style):
    """Adds appropriate light sources based on building style"""
    if building_style == "desert":
        light_block = Block("lantern")
        use_lantern = True
    elif building_style == "stone":
        light_block = Block("wall_torch", {"facing": "south"})
        use_lantern = False
    else:  # forest
        light_block = Block("wall_torch", {"facing": "south"})
        use_lantern = False

    wall_torch_x = x + width // 2
    if use_lantern:
        editor.placeBlock((wall_torch_x, y + 3, z + 1), light_block)
    else:
        editor.placeBlock((wall_torch_x, y + 2, z + 1), Block("wall_torch", {"facing": "south"}))

    if use_lantern:
        editor.placeBlock((wall_torch_x, y + 3, z + depth - 2), light_block)
    else:
        editor.placeBlock((wall_torch_x, y + 2, z + depth - 2), Block("wall_torch", {"facing": "north"}))

    wall_torch_z = z + depth // 2
    if use_lantern:
        editor.placeBlock((x + width - 2, y + 3, wall_torch_z), light_block)
    else:
        editor.placeBlock((x + width - 2, y + 2, wall_torch_z), Block("wall_torch", {"facing": "west"}))

    if use_lantern:
        editor.placeBlock((x + 1, y + 3, wall_torch_z), light_block)
    else:
        editor.placeBlock((x + 1, y + 2, wall_torch_z), Block("wall_torch", {"facing": "east"}))


def s4213211_generatePorch(editor, x, y, z, width):
    """Creates a porch on the front (north) side of the house"""
    porch_depth = 2
    placeCuboid(editor, (x, y, z - porch_depth), (x + width, y, z), [Block("oak_planks")])


def s4213211_generateGarden(editor, house_x, house_z, house_width, house_depth, x_min, z_min, x_max, z_max, heightmap):
    """Always creates a 4*4 garden near the house with flowers and fencing"""
    garden_size = 4
    garden_offset = 3

    sides = ["south", "east", "west", "north"]
    for side in sides:
        if side == "south":
            garden_x = house_x + house_width // 2 - garden_size // 2
            garden_z = house_z + house_depth + garden_offset
        elif side == "east":
            garden_x = house_x + house_width + garden_offset
            garden_z = house_z + house_depth // 2 - garden_size // 2
        elif side == "west":
            garden_x = house_x - garden_offset - garden_size
            garden_z = house_z + house_depth // 2 - garden_size // 2
        elif side == "north":
            garden_x = house_x + house_width // 2 - garden_size // 2
            garden_z = house_z - garden_offset - garden_size

        if (garden_x < x_min or garden_x + garden_size > x_max or
                garden_z < z_min or garden_z + garden_size > z_max):
            continue

        porch_depth = 2
        house_with_porch_z = house_z - porch_depth if side != "north" else house_z
        house_with_porch_depth = house_depth + porch_depth if side != "north" else house_depth

        if not s4213211_isOverlap(garden_x, garden_z, garden_size, garden_size,
                                  [(house_x, house_with_porch_z, house_width, house_with_porch_depth)]):
            break
    else:
        return None, None

    local_garden_x = garden_x - x_min
    local_garden_z = garden_z - z_min

    local_garden_x = max(0, min(local_garden_x, len(heightmap) - 1))
    local_garden_z = max(0, min(local_garden_z, len(heightmap[0]) - 1))

    garden_y = int(heightmap[local_garden_x, local_garden_z]) - 1

    for gx in range(garden_x, garden_x + garden_size):
        for gz in range(garden_z, garden_z + garden_size):
            local_gx = gx - x_min
            local_gz = gz - z_min
            local_gx = max(0, min(local_gx, len(heightmap) - 1))
            local_gz = max(0, min(local_gz, len(heightmap[0]) - 1))

            local_garden_y = int(heightmap[local_gx, local_gz]) - 1

            editor.placeBlock((gx, local_garden_y, gz), Block("grass_block"))
            flower_types = [
                Block("dandelion"),
                Block("poppy"),
                Block("blue_orchid"),
                Block("allium"),
                Block("azure_bluet"),
                Block("red_tulip"),
                Block("orange_tulip"),
                Block("white_tulip"),
                Block("pink_tulip"),
                Block("oxeye_daisy")
            ]
            editor.placeBlock((gx, local_garden_y + 1, gz), choice(flower_types))

    for gx in range(garden_x - 1, garden_x + garden_size + 1):
        local_gx = gx - x_min
        local_gz1 = (garden_z - 1) - z_min
        local_gz2 = (garden_z + garden_size) - z_min

        local_gx = max(0, min(local_gx, len(heightmap) - 1))
        local_gz1 = max(0, min(local_gz1, len(heightmap[0]) - 1))
        local_gz2 = max(0, min(local_gz2, len(heightmap[0]) - 1))

        fence_y1 = int(heightmap[local_gx, local_gz1]) - 1
        fence_y2 = int(heightmap[local_gx, local_gz2]) - 1

        editor.placeBlock((gx, fence_y1 + 1, garden_z - 1), Block("oak_fence"))
        editor.placeBlock((gx, fence_y2 + 1, garden_z + garden_size), Block("oak_fence"))

    for gz in range(garden_z, garden_z + garden_size):
        local_gx1 = (garden_x - 1) - x_min
        local_gx2 = (garden_x + garden_size) - x_min
        local_gz = gz - z_min

        local_gx1 = max(0, min(local_gx1, len(heightmap) - 1))
        local_gx2 = max(0, min(local_gx2, len(heightmap) - 1))
        local_gz = max(0, min(local_gz, len(heightmap[0]) - 1))

        fence_y1 = int(heightmap[local_gx1, local_gz]) - 1
        fence_y2 = int(heightmap[local_gx2, local_gz]) - 1

        editor.placeBlock((garden_x - 1, fence_y1 + 1, gz), Block("oak_fence"))
        editor.placeBlock((garden_x + garden_size, fence_y2 + 1, gz), Block("oak_fence"))

    return garden_x, garden_z

def s4213211_visualizeTerrain(heightmap, buildArea, output_dir="terrain_analysis"):
    """Creates terrain analysis visualizations for the suitability score calculation
       dependecies:numpy, matplotlib"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    heightmap_array = np.array(heightmap)

    gradient_x = np.gradient(heightmap_array, axis=0)
    gradient_z = np.gradient(heightmap_array, axis=1)

    slope_magnitude = np.sqrt(gradient_x ** 2 + gradient_z ** 2)

    suitability = 1.0 - np.clip(slope_magnitude / 3.0, 0, 1)

    colors = [(0.8, 0, 0), (0.8, 0.8, 0), (0, 0.8, 0)]
    cmap_name = 'suitability_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    plt.figure(figsize=(10, 8))
    suit_map = plt.imshow(suitability, cmap=cm)
    plt.colorbar(suit_map, label='Building Suitability (0-1)')
    plt.title('Building Location Suitability')
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.savefig(f"{output_dir}/building_suitability.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_range = np.arange(0, heightmap_array.shape[0])
    z_range = np.arange(0, heightmap_array.shape[1])
    X, Z = np.meshgrid(x_range, z_range)

    surf = ax.plot_surface(X, Z, heightmap_array.T, cmap='terrain',
                          linewidth=0, antialiased=True, alpha=0.8)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Height')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('Height')
    ax.set_title('3D Terrain Visualization')

    plt.savefig(f"{output_dir}/3d_terrain.png", dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "building_suitability": f"{output_dir}/building_suitability.png",
        "3d_terrain": f"{output_dir}/3d_terrain.png"
    }


def s4213211_visualizeBuildingLocation(heightmap, buildArea, house_x, house_z, house_width, house_depth,
                                       output_dir="terrain_analysis"):
    """Visualizes the selected building location on a terrain map"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_min = buildArea.offset.x
    z_min = buildArea.offset.z

    local_house_x = house_x - x_min
    local_house_z = house_z - z_min

    heightmap_array = np.array(heightmap)

    house_mask = np.zeros_like(heightmap_array)

    max_x = min(local_house_x + house_width, heightmap_array.shape[0])
    max_z = min(local_house_z + house_depth, heightmap_array.shape[1])

    for x in range(max(0, local_house_x), max_x):
        for z in range(max(0, local_house_z), max_z):
            if 0 <= x < heightmap_array.shape[0] and 0 <= z < heightmap_array.shape[1]:
                house_mask[x, z] = 1

    plt.figure(figsize=(10, 8))
    plt.imshow(heightmap_array, cmap='terrain', alpha=0.7)
    plt.imshow(house_mask, cmap='binary', alpha=0.3)

    x_coords = [local_house_x, local_house_x + house_width, local_house_x + house_width, local_house_x, local_house_x]
    z_coords = [local_house_z, local_house_z, local_house_z + house_depth, local_house_z + house_depth, local_house_z]
    plt.plot(x_coords, z_coords, 'r-', linewidth=2)

    plt.colorbar(label='Height')
    plt.title('Building Location on Terrain')
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.savefig(f"{output_dir}/building_location.png", dpi=300, bbox_inches='tight')
    plt.close()

    return f"{output_dir}/building_location.png"

# ---------------------------
# Main program
# ---------------------------
def main():
    global house_width
    editor = Editor(buffering=True)
    buildArea = editor.getBuildArea()
    editor.loadWorldSlice(cache=True)
    heightmap = editor.worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]

    # Create terrain analysis visualization before generating buildings
    print("Analyzing and visualizing terrain...")
    visualization_paths = s4213211_visualizeTerrain(heightmap, buildArea)
    print(f"Terrain visualizations saved to: {', '.join(visualization_paths.values())}")

    x_min = buildArea.offset.x
    z_min = buildArea.offset.z
    x_max = x_min + buildArea.size.x
    z_max = z_min + buildArea.size.z

    # House size parameters
    global house_width
    house_width = randint(6, 10)
    house_depth = randint(8, 12)

    floor_height = 3
    floor_count = randint(2, 4)
    house_height = floor_count * floor_height

    attempts = 0
    max_attempts = 30
    adaptive_house_width = house_width
    adaptive_house_depth = house_depth

    while attempts < max_attempts:
        candidate_x = randint(x_min, x_max - adaptive_house_width)
        candidate_z = randint(z_min, z_max - adaptive_house_depth)

        if s4213211_isOverlap(candidate_x, candidate_z, adaptive_house_width, adaptive_house_depth, existingStructures):
            attempts += 1

            if attempts > 20 and adaptive_house_width > 6 and adaptive_house_depth > 6:
                adaptive_house_width -= 1
                adaptive_house_depth -= 1
                print(f"Reducing house size to {adaptive_house_width}x{adaptive_house_depth} to avoid overlap")
                attempts = 15

            continue

        local_candidate_x = candidate_x - x_min
        local_candidate_z = candidate_z - z_min

        local_candidate_x = max(0, min(local_candidate_x, buildArea.size.x - 1))
        local_candidate_z = max(0, min(local_candidate_z, buildArea.size.z - 1))

        candidate_y = int(heightmap[local_candidate_x, local_candidate_z]) - 1

        if editor.getBlock((candidate_x, candidate_y, candidate_z)) == Block("water") or \
                editor.getBlock((candidate_x, candidate_y + 1, candidate_z)) == Block("water"):
            attempts += 1
            continue

        x = candidate_x
        z = candidate_z
        house_width = adaptive_house_width
        house_depth = adaptive_house_depth
        break
    else:
        print("Could not find non-overlapping position right now")
        print("Placing house at last attempted position, overlap may occur")
        x = candidate_x
        z = candidate_z
        house_width = adaptive_house_width
        house_depth = adaptive_house_depth

    local_x = x - x_min
    local_z = z - z_min
    y = int(heightmap[local_x, local_z]) - 1

    # Visualize the selected building location before construction
    building_location_visualization = s4213211_visualizeBuildingLocation(
        heightmap, buildArea, x, z, house_width, house_depth
    )
    print(f"Building location visualization saved to: {building_location_visualization}")

    # Generate adaptive foundation
    y = s4213211_adaptiveFoundation(editor, x, z, house_width, house_depth, heightmap, x_min, z_min)

    # Select building style based on location
    building_style = s4213211_selectBuildingStyle(editor, x, z, heightmap, x_min, z_min)
    print(f"Selected building style: {building_style}")

    # Get building materials
    materials = s4213211_getBuildingMaterials(building_style)
    wallBlock = materials["wall"]
    floorPalette = [materials["floor"]]
    roof_stair_type = materials["roof_stair"]
    roof_plank_type = materials["roof_plank"]

    print(f"Chosen wall block: {wallBlock}")
    print(f"Chosen roof stair type: {roof_stair_type}, roof plank type: {roof_plank_type}")

    # Generate house structure and decorations
    s4213211_generateHouse(editor, x, y, z, house_width, house_height, house_depth, wallBlock, floorPalette)
    existingStructures.append((x, z, house_width, house_depth, "house"))

    s4213211_generateRoof(editor, x, y, z, house_width, house_height, house_depth, roof_stair_type, roof_plank_type)
    s4213211_generateStyledWindows(editor, x, y, z, house_width, house_height, house_depth, building_style)
    s4213211_generateDoor(editor, x, y, z, house_width)
    s4213211_decorateInterior(editor, x, y, z, house_width, house_height, house_depth, building_style, materials)

    # Generate porch
    porch_depth = 2
    s4213211_generatePorch(editor, x, y, z, house_width)
    existingStructures.append((x, z - porch_depth, house_width, porch_depth, "porch"))

    # Generate garden with house parameters
    garden_size = 4
    garden_x, garden_z = s4213211_generateGarden(editor, x, z, house_width, house_depth, x_min, z_min, x_max, z_max,
                                                 heightmap)
    if garden_x is not None and garden_z is not None:
        existingStructures.append((garden_x, garden_z, garden_size, garden_size, "garden"))

    editor.flushBuffer()
    sys.exit()

if __name__ == "__main__":
    main()