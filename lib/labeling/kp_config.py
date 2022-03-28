import numpy as np
from matplotlib import pyplot as plt

# See README for details

# Keypoints for each shape-class
shape_class_kps = {
    "box_like": [
        "box_corner_front_tl",
        "box_corner_front_tr",
        "box_corner_front_br",
        "box_corner_front_bl",
        "box_corner_back_tl",
        "box_corner_back_tr",
        "box_corner_back_br",
        "box_corner_back_bl",
    ],
    "cylinder_like": [
        "cyl_top_center",
        "cyl_bottom_center",
        "cyl_rim_top_front",
        "cyl_rim_top_back",
        "cyl_rim_top_right",
        "cyl_rim_top_left",
        "cyl_rim_bottom_front",
        "cyl_rim_bottom_back",
        "cyl_rim_bottom_right",
        "cyl_rim_bottom_left",
    ],
    "hand_tool": [
        "tactile_point",
        "rotation_axis",
        "tool_base_front_left",
        "tool_base_front_right",
        "tool_base_back_left",
        "tool_base_back_right",
    ],
}

# Instance-specific keypoints
instance_shape_kps = {
    "grip": [
        "grip_thumb",
        "grip_palm",
        "grip_index",
        "grip_pinky",
    ],
    "spout": [
        "spout",
    ],
}

instance_texture_kps = {
    "brand_name": [
        "brand_name_tl",
        "brand_name_tr",
        "brand_name_br",
        "brand_name_bl",
    ],
    "nutrition_facts": [
        "nutrition_facts_tl",
        "nutrition_facts_tr",
        "nutrition_facts_br",
        "nutrition_facts_bl",
    ],
    "bar_code": [
        "bar_code_tl",
        "bar_code_tr",
        "bar_code_br",
        "bar_code_bl",
    ],
}

# The ordering of keypoints. Combine the lists to get the total ordering
shape_class_keys = ["box_like", "cylinder_like", "hand_tool"]
instance_shape_keys = ["grip", "spout"]
instance_texture_keys = ["brand_name", "nutrition_facts", "bar_code"]

# Combine into list of all keypoints so that we can define the channel order for prediction.
# Place the keypoints in the same order as above to make it easy to know the order (i.e. if
# we want to run the network in C++ or something).
kp_list = []
for k in shape_class_keys:
    kp_list += shape_class_kps[k]
for k in instance_shape_keys:
    kp_list += instance_shape_kps[k]
for k in instance_texture_keys:
    kp_list += instance_texture_kps[k]

# Quick error check
assert len(kp_list) == len(set(kp_list)), "Duplicate keypoint found in kp_list"

def num_kp():
    return len(kp_list)

# Get the BGR colors for all keypoints
np.random.seed(123456)
kp_cols = (255*plt.cm.get_cmap("gist_rainbow")
        (np.linspace(0, 1.0, num_kp()))[:, :3][:,::-1]).astype(np.int)
np.random.shuffle(kp_cols)
def kp_colors():
    return kp_cols

# Get the color of a single keypoint
def kp_color(kp_name):
    return kp_colors()[kp_list.index(kp_name)]

# Get a dict mapping keypoints to ID (index in above list)
def get_kps(class_str, has_grip, has_spout, has_brand_name, 
        has_nutrition_facts, has_bar_code):
    
    ret = {}
    assert class_str in shape_class_kps.keys(), \
            f"Shape class {class_str} is invalid! Options are {list(shape_class_kps.keys())}"
    for s in shape_class_kps[class_str]:
        ret[s] = kp_list.index(s)

    if has_grip:
        for s in instance_shape_kps["grip"]:
            ret[s] = kp_list.index(s)
    
    if has_spout:
        for s in instance_shape_kps["spout"]:
            ret[s] = kp_list.index(s)

    if has_brand_name:
        for s in instance_texture_kps["brand_name"]:
            ret[s] = kp_list.index(s)

    if has_nutrition_facts:
        for s in instance_texture_kps["nutrition_facts"]:
            ret[s] = kp_list.index(s)

    if has_bar_code:
        for s in instance_texture_kps["bar_code"]:
            ret[s] = kp_list.index(s)

    return ret


# Get the dict from above but given a pandas dataframe of the config file, and 
# object ID (starting from 1 as in BOP)
def load_kp_config(kp_config_df, object_id):
    kp_config = kp_config_df.iloc(0)[object_id-1]
    return get_kps(kp_config["class"], 
                   kp_config["has_grip"], 
                   kp_config["has_spout"], 
                   kp_config["has_brand_name"], 
                   kp_config["has_nutrition_facts"], 
                   kp_config["has_bar_code"])


# Inspect the keypoint colors
if __name__ == '__main__':
    import cv2
    img = np.zeros((100, 640, 3), dtype=np.uint8)

    colors = kp_colors()
    print("Num kp: ", num_kp())
    dx = img.shape[1] // len(colors)
    for i in range(len(colors)):
        cv2.circle(img, (dx + i*dx, img.shape[0]//2), 10, colors[i].tolist(), -1)

    cv2.imshow("Keypoint colors", img)
    cv2.waitKey(0)
