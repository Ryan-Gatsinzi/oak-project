

line_old = ['0\t0\t0\t0\t0\t0\t0\t0\t9\t', '806\t799\t803\t806\t802\t799\t789\t797\t20\t', '806\t799\t803\t806\t802\t799\t789\t797\t29\t', '806\t7']
line = ['29\t', '806\t799\t803\t806\t802\t799\t789\t796\t39\t', '806\t799\t803\t806\t802\t799\t789\t796\t49\t', '806\t799\t803\t806\t802']
line_byte = b'29\t\n806\t799\t803\t806\t802\t799\t789\t796\t39\t\n806\t799\t803\t806\t802\t799\t789\t796\t49\t\n806\t799\t803\t806\t802\t'
line_str = str(line_byte)
line1 = line_str.lstrip("\n")

list_test = [0]*10
line2 = line_byte.decode().rstrip().split('\n')
for i in line2:
    if len(i) == 35:
        data_line = i
        data_array = data_line.split('\t')
        break
    print(len(i))

print(data_array)