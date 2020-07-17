file_box = open('./list_bbox_celeba.txt', 'r')
file_attr = open('./list_attr_celeba.txt', 'r')

box_lines = file_box.readlines()[2:]
attr_lines = file_attr.readlines()[2:]

new_lines = []
for i in range(180000, 200000-1):
    imgId_box = box_lines[i]
    attr_line = attr_lines[i].strip().split()
    new_line = imgId_box.strip() + ' ' + attr_line[-9]
    new_lines.append(new_line)
    if i % 1000 == 0:
        print(str(i))

with open('val_labels.txt', 'w') as attr_file:
    for line in new_lines:
        attr_file.write(line + '\n')
