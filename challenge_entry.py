from tank_generator import TankGenerator
from textwrap import wrap

class ChallengeEntry:
    '''2019 /r/proceduralgeneration Armoured Fighting Vehicles challenge entry'''

    @classmethod
    def make_some_tanks(cls, count=1, output_file='A Brief History of Tanks.md', preview=False):
        '''Generate the names and descriptions of tanks and save them to a markdown file'''

        # if GPT2 has been preloaded, use it
        if 'gpt2_module' in globals():
            gen = TankGenerator(globals('gpt2_module'))
        else:
            gen = TankGenerator()
        tanks = []

        while len(tanks) < count:
            name = gen.tank_name()
            # name can be None if the generator keeps rolling names that fail the constraints
            if name == None or name in [tank['name'] for tank in tanks]:
                continue

            tanks.append({
                'name': name,
                'description': gen.tank_description(name)
            })

        blocks = []
        for tank in tanks:
            blocks.append('\n\n'.join(['## ' + tank['name'].upper(), tank['description']]))
            if preview:
                print(f'## {tank["name"].upper()}\n')
                for line in wrap(tank['description'], 80): print(line)
                print('')

        output = '# A Brief History of Tanks\n\n'
        output += '\n\n'.join(blocks)
        if output_file != None:
            with open(output_file, 'w') as f:
                f.write(output)