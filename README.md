# Image Morphing

Final project for CS/EE 166 (Computational Cameras)

Presentation Link: [https://www.youtube.com/watch?v=Ul1cwGuRIn8](https://www.youtube.com/watch?v=Ul1cwGuRIn8)

![Cat Morphing](output/blep.gif)

## Usage

```
>>> ./morph.py -h
usage: morph.py [-h] [--task {interp_images,interp_meshes,draw_mesh}] [--nframes NFRAMES] [--fname FNAME]
                src_img dst_img src_mesh dst_mesh

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

## File Structure

```
.
├── morph.py            Self-contained script, see above
├── notebook.py         Generates visualization for presentation
├── frames              Generated frames of output gif
├── input               Input images (png)
├── meshes              Input meshes (csv)
├── output              Output (gif / png)
├── report              Project proposal and report
├── slides              Slides, made with reveal.js
└── visuals             Visualizations for slides and and report
```
