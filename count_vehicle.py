from xml.dom import minidom

# parse an xml file by name
mydoc = minidom.parse('result.rou.xml')

flows = mydoc.getElementsByTagName('flow')

# all item attributes
count = 0
print('\nAll attributes:')  
for elem in flows:
    count += int(elem.attributes['number'].value)

print count