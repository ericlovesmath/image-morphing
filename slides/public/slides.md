## Image Morphing with Delaunay Triangulation

CS/EE 166 Final Project

[Eric Lee](https://www.ericchanlee.com/), Caltech CS

---

# Delaunay Triangulation

---

## Naive Cross Fade

<div class="container">

<img class="col" src="./imgs/idina_menzel.png" width="20%" />
<img class="col" src="./imgs/jamie_muscato.png" width="20%" />
<img class="col" src="./imgs/cross_fade.gif" width="20%" />

</div>

<!--
Explain actors
Goal: Do better than this
-->

---

## Triangulation

<div class="container">

<img class="col" src="./imgs/idina_menzel_mesh.png" width="40%" />
<img class="col" src="./imgs/jamie_muscato_mesh.png" width="40%" />

</div>

<!--
Avoids having to select triangles, just lets you pick points
So how do we triangulate?
-->

---

## Delaunay Triangulation

<img src="./imgs/delaunay_dual.png" width="70%"/>

Convex Hull division

<!--
Takes plane, subdivides their convex hull into triangles whose
circumcircles do not contain any of the verticies.
This maximizes the size of the smallest angle in any of the triangles.
Useful to avoid sliver simplexs (sharp triangles)

Dual to Voronoi Diagram.

Iterative algorithm described in original paper. Bowyer–Watson algorithm
Convex Hull Optimization tools exist though.

[Previous slide to show how nice everything looks]

But triangles don't correspond
-->


---

## Average Mesh

<img src="./imgs/average_mesh.png" />

<!--
We triangulate the average of the points and map those same triangles back
-->

---

## Remap Average Mesh

<div class="container">

<img class="col" src="./imgs/average_idina_menzel_mesh.png" width="40%" />
<img class="col" src="./imgs/average_jamie_muscato_mesh.png" width="40%" />

</div>

<!--
Observe "Pull" of triangles is visible and clear
-->

---

# Image Morphing

---

## Simplex Mapping

<img src="./imgs/single_triangle_transform.gif" />

<!--
Linear Map
Triangle is 2D but transformation is affine linear in 2D so 3D matrix is required
-->

---

## Simplex Mapping

$$
\begin{pmatrix}
    x_1' & x_2' & x_3' \\\\
    y_1' & y_2' & y_3' \\\\
    1 & 1 & 1
\end{pmatrix}
\begin{pmatrix}
    x_1 & x_2 & x_3 \\\\
    y_1 & y_2 & y_3 \\\\
    1 & 1 & 1
\end{pmatrix}^{-1}
$$

Transforming $T$ to $T'$

<!--
Trick: Treat each vertex as extra z = 1
Change of basis
-->

---

## Simplex Mapping Example

<img src="./imgs/triangle_transform.gif" />

<!--
Applied to each frame of our original actors
Problem: This is continuous. We want discrete pixels colors.
-->

---

## Inverse Color Sampling

<img src="./imgs/sampling.png" />

<!--
Inverse sampling to avoid gaps in sampling
Bilinear interpolation
-->

---

## Color Sampling Example

<div class="container">

<img class="col" src="./imgs/inv_sample_menzel.png" width="40%" />
<img class="col" src="./imgs/inv_sample_muscato.png" width="40%" />

</div>

$\alpha = 0.4$, Check simplex with Barycentric Coordinates

<!--
Barycentric is just (0,0,1) type for each corner
More efficient than other algs?
Do this for each pixel
This is taking a frame at a = 0.4 then seeing where each pixel would sample
Observe big gaps in Muscato
-->

---

## Menzel -> Muscato

<img src="./imgs/final_transform.gif" width="50%"/>

<!--
Ear
-->

---

## Brady -> Jesse Eisenberg

<div class="container">

<img class="col" src="./imgs/brady.png" width="30%" />
<img class="col" src="./imgs/jesse_eisenberg.png" width="30%" />
<img class="col" src="./imgs/bradenberg.gif" width="30%" />

</div>

<!--
Neck
-->

---

## Here kitty kitty kitty kitty kitty kitty kitty kitty

<div class="container">

<img class="col" src="./imgs/blep1.png" width="20%" />
<img class="col" src="./imgs/blep2.png" width="20%" />
<img class="col" src="./imgs/blep3.png" width="20%" />
<img class="col" src="./imgs/blep4.png" width="20%" />
<img class="col" src="./imgs/blep5.png" width="20%" />

</div>

Source: [r/blep](https://www.reddit.com/r/Blep/top/)

<!--
Diff angles of faces and stuff
Tongue was necessary for anchor
-->

---

<img src="./imgs/blep.gif" width="50%" />

(Simple mesh on faces, eyes, and blep only)

---

## Mesh Morphing

---

<img src="./imgs/scary_idina.gif" width="50%"/>

Only sampling colors from initial source image

<!--
Scary
-->

---

## Flower Transformation

<div class="container">

<img class="col" src="./imgs/flower.png" width="30%" />
<img class="col" src="./imgs/flower_rot.gif" width="30%" />
<img class="col" src="./imgs/flower_close.gif" width="30%" />

</div>

<!--
Animation Application
Shrinking mesh is smaller, rot mesh is bigger
Nice grooves
-->


---

## Caricature

<div class="container">

<img class="col" src="./imgs/jesse_eisenberg.png" width="30%" />
<img class="col" src="./imgs/caricuture.png" width="30%" />
<img class="col" src="./imgs/final_caricature.png" width="30%" />

</div>

---

# Demonstration

```
>>> ./morph.py -h
usage: morph.py [-h] [--task {interp_images,interp_meshes,draw_mesh}] [--nframes NFRAMES] [--fname FNAME] src_img dst_img src_mesh dst_mesh

Draw meshes and interpolate images.

positional arguments:
  src_img               Path to image used as source
  dst_img               Path to image used as destination
  src_mesh              Path to mesh corresponding to src_img
  dst_mesh              Path to mesh corresponding to dst_img

options:
  -h, --help            show this help message and exit
  --task {interp_images,interp_meshes,draw_mesh}
                        interp_images: Interpolate between images (default).
                        interp_meshes: Ignore dst_img and sample interpolation from only
                                       src_img, effectively animating the image.
                        draw_mesh: Select points on image to draw mesh, ignores dst_* args.
  --nframes NFRAMES     Number of frames to generate (default: 10)
  --fname FNAME         Name of generated gif and frames (default: 'result')
```

---

### Extensions / Limitations

- Forward sampling could be faster (~10 sec per frame now)
- Delaunay triangulation on average mesh could create slivers
- Selecting points could be done automatically through ML algorithms
- Simpler than other algorithms (Beier-Neely)

<!--
Beier-Neely warps lines matching instead of point meshes matching
-->

---

# Questions?

<img src="./imgs/blep.gif" width="40%" />

[https://github.com/ericlovesmath/image-morphing](https://github.com/ericlovesmath/image-morphing)
