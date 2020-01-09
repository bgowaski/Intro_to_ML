# Read the image file.
from scipy import misc
data = misc.imread('1.jpg')

# Make a string with the format you want.
text = ''
for row in data:
    for e in row:
        text += '{} {} {}'.format(e[0], e[1], e[2])
    text += '\n'

# Write the string to a file.
with open('1.txt', 'w') as f:
    f.write(text)