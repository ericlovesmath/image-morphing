./morph.py imgs/idina_menzel.png imgs/jamie_muscato.png meshes/idina_menzel.csv meshes/jamie_muscato.csv --nframes 40
./morph.py imgs/idina_menzel.png imgs/jamie_muscato.png meshes/idina_menzel.csv meshes/jamie_muscato.csv --nframes 40 --task intrep_meshes

./morph.py imgs/flower.png - meshes/flower.csv - --task draw_mesh
./morph.py imgs/flower.png - meshes/flower_close.csv - --task draw_mesh
./morph.py imgs/flower.png - meshes/flower.csv meshes/flower_close.csv --task interp_meshes --fname flower_close --nframes 40

./morph.py imgs/flower.png - meshes/flower_dense.csv - --task draw_mesh
./morph.py imgs/flower.png - meshes/flower_rot.csv - --task draw_mesh
./morph.py imgs/flower.png - meshes/flower_dense.csv meshes/flower_rot.csv --task interp_meshes --fname flower_rot --nframes 40

./morph.py imgs/blep1.png - meshes/blep1.csv - --task draw_mesh
./morph.py imgs/blep2.png - meshes/blep2.csv - --task draw_mesh
./morph.py imgs/blep3.png - meshes/blep3.csv - --task draw_mesh
./morph.py imgs/blep4.png - meshes/blep4.csv - --task draw_mesh
./morph.py imgs/blep5.png - meshes/blep5.csv - --task draw_mesh

./morph.py imgs/blep1.png imgs/blep2.png meshes/blep1.csv meshes/blep2.csv --fname blep12 --nframes 30
./morph.py imgs/blep2.png imgs/blep3.png meshes/blep2.csv meshes/blep3.csv --fname blep23 --nframes 30
./morph.py imgs/blep3.png imgs/blep4.png meshes/blep3.csv meshes/blep4.csv --fname blep34 --nframes 30
./morph.py imgs/blep4.png imgs/blep5.png meshes/blep4.csv meshes/blep5.csv --fname blep45 --nframes 30
./morph.py imgs/blep5.png imgs/blep1.png meshes/blep5.csv meshes/blep1.csv --fname blep51 --nframes 30

./morph.py imgs/brady.png - meshes/brady.csv - --task draw_mesh
./morph.py imgs/jesse_eisenberg.png - meshes/jesse_eisenberg.csv - --task draw_mesh
./morph.py imgs/caricuture.png - meshes/caricuture.csv - --task draw_mesh
./morph.py imgs/brady.png imgs/jesse_eisenberg.png meshes/brady.csv meshes/jesse_eisenberg.csv --fname bradenberg --nframes 40
./morph.py imgs/jesse_eisenberg.png imgs/caricuture.png meshes/jesse_eisenberg.csv meshes/caricuture.csv --task interp_meshes --fname caricuture_eisenberg --nframes 20