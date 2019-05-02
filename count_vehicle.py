from xml.dom import minidom

# parse an xml file by name
rou_EW = minidom.parse('routes/result.rou_EW.xml')
rou_NS = minidom.parse('routes/result.rou_NS.xml')
rou_NS_EW = minidom.parse('routes/result.rou_NS_EW.xml')
rou_less = minidom.parse('routes/result.rou_less.xml')

count_vehicle = []
for i in range(4):
    count_vehicle.append(0)

# LOW routes
flows = rou_less.getElementsByTagName('flow')
for elem in flows:
    count_vehicle[0] += int(elem.attributes['number'].value)

# HIGH routes
flows = rou_NS_EW.getElementsByTagName('flow')
for elem in flows:
    count_vehicle[1] += int(elem.attributes['number'].value)

# NS routes
flows = rou_NS.getElementsByTagName('flow')
for elem in flows:
    count_vehicle[2] += int(elem.attributes['number'].value)

# EW routes
flows = rou_EW.getElementsByTagName('flow')
for elem in flows:
    count_vehicle[3] += int(elem.attributes['number'].value)

print (count_vehicle)