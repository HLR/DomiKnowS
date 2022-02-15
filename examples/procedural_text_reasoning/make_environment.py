file = open('environment.txt', 'r')
lines = file.readlines()

store_file = open('domiknows_environment.txt', 'w')


for line in lines:
    each_line = line.split('	')
    generate_text = 'pip install ' + each_line[0] + '==' + each_line[1]
    store_file.writelines(generate_text + '\n')

file.close()
store_file.close()