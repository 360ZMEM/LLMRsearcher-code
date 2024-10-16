# You can specify the requirements in which the reward component is generated in more detail here. For example, 'collision' and 'bordercrossing' can be merged into a single requirement or split. These requirements and descriptions are conducted as a dict.
# objectives = ['collision','bordercrossing','data_overflow','energy'] # in order

desc_dict ={
        'collision':'(One of safety requirement) The number of both **collisions** should be **reduced to zero**.',
        'border_crossing':'(One of safety requirement) The number of both **collisions** should be **reduced to zero**.',
        'data_overflow':'The number of data overflows **should be reduced to zero as much as possible**. This can be achieved by **responding promptly** to SNs and **increasing** the number of served SNs.',
        'energy': 'The energy consumption of AUVs may be optimized (lower is better) without violating other requirements(if possible).'
}
# objectives <==> training log
obj_log_dict = {
        'collision':['collisions'], 'border_crossing': ['crossing the border'], 
        'data_overflow':['data overflow', 'total served SNs'], 
        'energy':['energy consumption']
}

objectives = list(desc_dict.keys())

