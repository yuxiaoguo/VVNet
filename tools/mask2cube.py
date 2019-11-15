import sys
import struct
import numpy as np


ply_cube_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
element face %d
property list uchar int vertex_index
end_header
%s"""


def mask2cube(src, des):
  assert(src.endswith('.mask'))
  with open(src, "rb") as f:
    size = struct.unpack("iii", f.read(4*3))

    points_raw = f.read(np.prod(size)*4)
    points = np.asarray(struct.unpack("i"*np.prod(size), points_raw), dtype=int).reshape(size[2], size[1], size[0])

  n = np.sum(points)

  with open(des, "w") as f:
    f.write(ply_cube_template % (8 * n, 6 * n, ''))

    shift = 0.5
    for z in range(size[2]):
      for y in range(size[1]):
        for x in range(size[0]):
          if points[z][y][x] == 1:
            f.write('\n%f %f %f' % (x - shift, y - shift, z - shift))
            f.write('\n%f %f %f' % (x - shift, y + shift, z - shift))
            f.write('\n%f %f %f' % (x + shift, y + shift, z - shift))
            f.write('\n%f %f %f' % (x + shift, y - shift, z - shift))
            f.write('\n%f %f %f' % (x - shift, y - shift, z + shift))
            f.write('\n%f %f %f' % (x - shift, y + shift, z + shift))
            f.write('\n%f %f %f' % (x + shift, y + shift, z + shift))
            f.write('\n%f %f %f' % (x + shift, y - shift, z + shift))
            f.write('\n')

    counter = 0
    for i in range(n):
      f.write("4 %d %d %d %d\n" % (counter, counter + 1, counter + 2, counter + 3))
      f.write("4 %d %d %d %d\n" % (counter, counter + 4, counter + 5, counter + 1))
      f.write("4 %d %d %d %d\n" % (counter, counter + 3, counter + 7, counter + 4))
      f.write("4 %d %d %d %d\n" % (counter + 6, counter + 5, counter + 4, counter + 7))
      f.write("4 %d %d %d %d\n" % (counter + 6, counter + 7, counter + 3, counter + 2))
      f.write("4 %d %d %d %d\n" % (counter + 6, counter + 2, counter + 1, counter + 5))
      counter += 8


def main():
  src = sys.argv[1]
  des = src.replace(".mask", "_mask.ply")
  mask2cube(src, des)


if __name__ == '__main__':
  main()
  """ usage: python mask2cube.py 0000.points"""