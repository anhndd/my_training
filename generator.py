from __future__ import print_function


def get_simu_type_str(simu_type):
    if simu_type == 0:
        return 'LOW'
    elif simu_type == 1:
        return 'HIGH'
    elif simu_type == 2:
        return 'NS'
    elif simu_type == 3:
        return 'EW'

def gen_route(episode):
    # load file:

    if episode % 4 == 0:
        with open("routes/sumoconfig.sumoconfig", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>
            <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                <input>
                    <net-file value="net.net.xml"/>
                        """, file=routes)
            print("""<route-files value="myroutes.rou_less.xml"/>""", file=routes)
            print("""    </input>
                <time>
                    <begin value="0"/>
                    <end value="20000"/>
                </time>
                
                <processing>
	            <lateral-resolution value="0.875"/>
                </processing>
                
                <report>
                    <xml-validation value="never"/>
                    <duration-log.disable value="true"/>
                    <no-step-log value="true"/>
                </report>
            </configuration> """, file=routes)
        return 0
    elif episode % 4 == 1:
        with open("routes/sumoconfig.sumoconfig", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>
            <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                <input>
                    <net-file value="net.net.xml"/>
                        """, file=routes)
            print("""<route-files value="myroutes.rou_NS_EW.xml"/>""", file=routes)
            print("""    </input>
                <time>
                    <begin value="0"/>
                    <end value="20000"/>
                </time>
                
                <processing>
	            <lateral-resolution value="0.875"/>
                </processing>
                
                <report>
                    <xml-validation value="never"/>
                    <duration-log.disable value="true"/>
                    <no-step-log value="true"/>
                </report>
            </configuration> """, file=routes)
        return 1
    elif episode % 4 == 2:
        with open("routes/sumoconfig.sumoconfig", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>
            <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                <input>
                    <net-file value="net.net.xml"/>
                        """, file=routes)
            print("""<route-files value="myroutes.rou_NS.xml"/>""", file=routes)
            print("""    </input>
                <time>
                    <begin value="0"/>
                    <end value="20000"/>
                </time>
                
                <processing>
	            <lateral-resolution value="0.875"/>
                </processing>
                
                <report>
                    <xml-validation value="never"/>
                    <duration-log.disable value="true"/>
                    <no-step-log value="true"/>
                </report>
            </configuration> """, file=routes)
        return 2
    elif episode % 4 == 3:
        with open("routes/sumoconfig.sumoconfig", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>
            <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                <input>
                    <net-file value="net.net.xml"/>
                        """, file=routes)
            print("""<route-files value="myroutes.rou_EW.xml"/>""", file=routes)
            print("""    </input>
                <time>
                    <begin value="0"/>
                    <end value="20000"/>
                </time>
                
                <processing>
	            <lateral-resolution value="0.875"/>
                </processing>
                
                <report>
                    <xml-validation value="never"/>
                    <duration-log.disable value="true"/>
                    <no-step-log value="true"/>
                </report>
            </configuration> """, file=routes)
        return 3
    # <route-files value="myroutes.rou_NS.xml"/>
    # <route-files value="myroutes.rou_EW.xml"/>
    # myroutes.rou_less
    # myroutes.rou_NS_EW.


''' 
<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="net.net.xml"/>
        <route-files value="myroutes.rou_EW.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="20000"/>
    </time>

    <processing>
	<lateral-resolution value="0.875"/>
    </processing>

    <report>
        <xml-validation value="never"/>
        <duration-log.disable value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
'''
