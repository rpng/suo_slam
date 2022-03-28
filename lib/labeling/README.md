# Keypoint Meanings

## Shape Class-Specific Keypoints
We have divided the YCB-Video objects into different basic shape classes as such.

```
          CLASS            INSTANCE
          
object -- box_like      -- cracker_box          
                           sugar_box
                           pudding_box
                           gelatin_box
                           wood_block
                           foam_brick
                           potted_meat_can
                           mustard_bottle
                           bleach_cleanser

       -- cylinder_like -- pitcher_base
                           mug
                           master_chef_can
                           tomato_soup_can
                           tuna_fish_can
                           bowl
                           large_marker
                           banana 

       -- hand_tool     -- power_drill
                           scissors
                           large_clamp
                           extra_large_clamp

```

Each shape class has a basic set of keypoints which define the shape of the object
as well as provide some basic semantic information about the object, such as the 
front, back, top, bottom, etc. This would not be very useful if all the CAD models
were in the canonical frame, but this is not the case for all BOP objects.

Many of the keypoints are labeled as "front", "back", "top", "bottom", etc.
From here on out, these labels should be human-identifyable unless the object 
has symmetries or an unusual nature which prevents this. For example, for grocery
items, the front should display the main logo, and the top and bottom should be such
that the text is oriented correctly to read. For tools, mug, pitcher, etc. or anything
else with a grip and no obvious front, then the front should be the face facing
someone who is holding the object with their right hand, and their fingers pointing
to the left. For tools, the tacile point should be towards the top.

Now here is the list of keypoints for each shape class:

### box_like
All eight corners of box going from front: top-left, top-right, bottom-right,
bottom-left, then rotate with 180 degrees of yaw and do the same thing.
- `box_corner_front_tl`
- `box_corner_front_tr`
- `box_corner_front_br`
- `box_corner_front_bl`
- `box_corner_back_tl`
- `box_corner_back_tr`
- `box_corner_back_br`
- `box_corner_back_bl`

### cylinder_like
Top center then bottom center of the circle-ish top and bottom of the main cylinder shape.
Additionally, 8 points around the edge of the top and bottom circles for the front, back, 
left and right. The "right" and "left" points are taken with respect to the front of the object.
- `cyl_top_center`
- `cyl_bottom_center`
- `cyl_rim_top_front`
- `cyl_rim_top_back`
- `cyl_rim_top_right`
- `cyl_rim_top_left`
- `cyl_rim_bottom_front`
- `cyl_rim_bottom_back`
- `cyl_rim_bottom_right`
- `cyl_rim_bottom_left`

### hand_tool
opening for the drill bit. Now, given the definition of the "front" of these objects
given above, we define four points on the "base" of the object which is on the bottom,
which are similar to the box corners. For the base points on the "back", like the box,
turn the object 180 degrees and then consider the left and right from this angle.
In addition to these points, there should be another point of interest named "rotation_axis"
that lies between the grip and the tactile point. For the clamps and scissors, make this the
axis of rotation, and for the drill, it should be on the end of the drilling axis opposite
of the tactile point.
- `tactile_point`
- `rotation_axis`
- `tool_base_front_left`
- `tool_base_front_right`
- `tool_base_back_left`
- `tool_base_back_right`

## Instance-Specific Keypoints

Some instances have shared defining features which are easily human-identifiable.
These type of keypoints should also be useful for a higher level task besides pose
estimation, such as reading the brand name of a product, or finding the
gripping point that a human would typically use. We have also placed these keypoints
into the categories of shape-based and texture-based keypoints. Shape-based keypoints
can be identified without any texture, and the texture-based ones rely on the texture
of the CAD model to identify.

## Shape-Based

### grip
Many objects have an obvious way to grip them. These keypoints should be used in this case.
Even though these grip points are labeled with fingers, this is just for convenience
of naming. The points should be exaggerated to the edge of what can be considered the grip.
Note that thumb and palm should be on the "right" and index and pinky on the "left" in ambiguous
cases. Make sure these line up with other keypoint labels as well which define 
front/back/left/right.
- `grip_thumb`
- `grip_palm`
- `grip_index`
- `grip_pinky`

### spout
This is any single point from which liquid should be poured or squeezed from the object.
This includes bottle caps, and the spout of a pitcher.
- `spout`

## Texture-Based

### brand_name
Brand name is considered the biggest brand name on the "front" of the object, where the 
"front" has been described above. Make the keypoints surround all of the text of the 
brand name with a minimal bounding quadrilateral such that all the text is inside.
- `brand_name_tl`
- `brand_name_tr`
- `brand_name_br`
- `brand_name_bl`

### nutrition_facts
Nutrition facts are usually in an identifiable black-and-white box. Orient the object
so that you can read them, then pick the four points from top-left, top-right,
bottom-right, bottom-left.
- `nutrition_facts_tl`
- `nutrition_facts_tr`
- `nutrition_facts_br`   
- `nutrition_facts_bl`    

### bar_code
Many store-bought items have bar codes. Find the biggest one, and consider the "top"
of the bar code as the top if you were to flip it around to read the numbers under the
bar code lines.
- `bar_code_tl`
- `bar_code_tr`
- `bar_code_br`
- `bar_code_bl`

