# Example usage

./morph.py input/idina_menzel.png input/jamie_muscato.png meshes/idina_menzel.csv meshes/jamie_muscato.csv --nframes 40
./morph.py input/idina_menzel.png input/jamie_muscato.png meshes/idina_menzel.csv meshes/jamie_muscato.csv --nframes 40 --task intrep_meshes

./morph.py input/flower.png - meshes/flower.csv - --task draw_mesh
./morph.py input/flower.png - meshes/flower_close.csv - --task draw_mesh
./morph.py input/flower.png - meshes/flower.csv meshes/flower_close.csv --task interp_meshes --fname flower_close --nframes 40

./morph.py input/flower.png - meshes/flower_dense.csv - --task draw_mesh
./morph.py input/flower.png - meshes/flower_rot.csv - --task draw_mesh
./morph.py input/flower.png - meshes/flower_dense.csv meshes/flower_rot.csv --task interp_meshes --fname flower_rot --nframes 40

./morph.py input/blep1.png - meshes/blep1.csv - --task draw_mesh
./morph.py input/blep2.png - meshes/blep2.csv - --task draw_mesh
./morph.py input/blep3.png - meshes/blep3.csv - --task draw_mesh
./morph.py input/blep4.png - meshes/blep4.csv - --task draw_mesh
./morph.py input/blep5.png - meshes/blep5.csv - --task draw_mesh

./morph.py input/blep1.png input/blep2.png meshes/blep1.csv meshes/blep2.csv --fname blep12 --nframes 30
./morph.py input/blep2.png input/blep3.png meshes/blep2.csv meshes/blep3.csv --fname blep23 --nframes 30
./morph.py input/blep3.png input/blep4.png meshes/blep3.csv meshes/blep4.csv --fname blep34 --nframes 30
./morph.py input/blep4.png input/blep5.png meshes/blep4.csv meshes/blep5.csv --fname blep45 --nframes 30
./morph.py input/blep5.png input/blep1.png meshes/blep5.csv meshes/blep1.csv --fname blep51 --nframes 30

./morph.py input/brady.png - meshes/brady.csv - --task draw_mesh
./morph.py input/jesse_eisenberg.png - meshes/jesse_eisenberg.csv - --task draw_mesh
./morph.py input/caricuture.png - meshes/caricuture.csv - --task draw_mesh
./morph.py input/brady.png input/jesse_eisenberg.png meshes/brady.csv meshes/jesse_eisenberg.csv --fname bradenberg --nframes 40
./morph.py input/jesse_eisenberg.png input/caricuture.png meshes/jesse_eisenberg.csv meshes/caricuture.csv --task interp_meshes --fname caricuture_eisenberg --nframes 20
