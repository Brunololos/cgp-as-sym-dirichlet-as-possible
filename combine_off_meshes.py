import sys

# receive input file
file_names = ["./meshes/RyanHolmesLetters/A.off", 
              "./meshes/RyanHolmesLetters/S.off", 
              "./meshes/RyanHolmesLetters/D.off", 
              "./meshes/RyanHolmesLetters/A.off", 
              "./meshes/RyanHolmesLetters/P_remesh.off"]

num_verts = [0]
num_edges = [0]
num_faces = [0]
files = []

for file_name in file_names:
    # check if input file is a .off file
    if file_name.endswith(".off"):
        name = file_name.split(".")[0]

    # open input file
    with open(file_name) as fp:
        fp.readline()
        line = fp.readline().split(' ')
        num_verts += [num_verts[-1] + int(line[0])]
        num_edges += [num_edges[-1] + int(line[1])]
        num_faces += [num_faces[-1] + int(line[2])]
        files += [fp.readlines()]

# create a new file
output_off = open("./meshes/documentation/combined.off", 'w')
output_off.write("OFF\n")
output_off.write(str(num_verts[-1]) + " " + str(num_edges[-1]) + " " + str(num_faces[-1]) + "\n")

# merge vertex lists
i = 0
for file in files:
    vert_avg = [0, 0, 0]
    for line in file:
        coords = line.split(' ')
        if len(coords) > 3:
            break
        coords_offset = [-4*(len(file_names)/2) + i*4, 0, 0]
        output_off.write(str(float(coords[0]) + coords_offset[0]) + " " + str(float(coords[1]) + coords_offset[1]) + " " + str(float(coords[2]) + coords_offset[2]) + "\n")
        vert_avg = [vert_avg[0] + float(coords[0]), vert_avg[1] + float(coords[1]), vert_avg[2] + float(coords[2])]
    print("vert_avg:")
    print(vert_avg[0] / float(num_verts[i + 1] - num_verts[i]))
    print(vert_avg[1] / float(num_verts[i + 1] - num_verts[i]))
    print(vert_avg[2] / float(num_verts[i + 1] - num_verts[i]))
    i += 1

# merge face lists
i = 0
for file in files:
    index_offset = num_verts[i]
    line_idx = num_verts[i+1] - num_verts[i]
    max_line_idx = line_idx + num_edges[i+1] - num_edges[i]
    #print(line_idx)
    for j in range(line_idx, max_line_idx):
        line = file[j]
        face = line.split('\n')[0].split(' ')
        #print(face)
        output_off.write(face[0])
        for k in range(1, int(face[0]) + 1):
            output_off.write(" " + str(int(face[k]) + index_offset))
        output_off.write("\n")
    i += 1
print(num_verts)
print(num_edges)
print(num_faces)
output_off.close()