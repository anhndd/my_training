import count_vehicle

memory_size = 20000								# 20.000
mini_batch_size = 64							# 64
a_dec = 4.5
num_of_phase = 2
action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]  
idLightControl = '4628048104'



# ==============
Num_Exploration = 2000						# exploring in: 2.000 steps.
Num_Training	= 30000						# training in: 30.000 steps
Num_Testing     = 250000

count_vehicle = count_vehicle.count_vehicle

# testing:
# Num_Exploration = 10						# exploring in: 2.000 steps.
# mini_batch_size = 4
# memory_size = 50