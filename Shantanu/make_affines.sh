#/bin/bash

# Affine transform by 7 degrees both ways

for d in data/kanji-*; do
  echo "Starting with $d"
  for f in $d/*; do
    filename=$(basename "$f")
    filename="${filename%.*}"
    convert "${f}" +distort AffineProjection "1,-0.122173,0,1,0,0" +repage "$d/${filename}_affine_left.png"
    convert "${f}" +distort AffineProjection "1,0.122173,0,1,0,0" +repage "$d/${filename}_affine_right.png"
  done
done
