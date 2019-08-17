import random, re
from generic_generator import GenericGenerator

class TankGenerator(GenericGenerator):
    '''Generate tank names and descriptions using GPT-2'''

    def tank_name(self):
        ''''Generates and returns a single tank name'''

        # these are the requirements for an acceptable tank name
        def constraints(text):
            if len(text) < 3 or len(text) > 18: return False
            if re.search(r'[^A-zÀ-ú\d\s\-]', text) != None: return False
            if text[0] != text[0].upper(): return False
            if text.lower().startswith('the'): return False
            if text.lower().endswith('tank'): return False
            if text.lower().endswith('tanks'): return False
            if text.endswith('-'): return False
            return True

        prompt = 'The name of this armored tank is "'
        # this expects GPT-2 to fill in the blank between the quotes
        return self.generate(prompt, '"', constraints=constraints, max_tokens=8)
    
    def tank_description(self, tank_name):
        '''Generates a single paragraph description of the given tank name'''

        # pronoun generator for beginning of sentences
        def it():
            return random.choice([
                'It',
                'The tank',
                f'The {tank_name}',
                f'The {tank_name} tank'
            ])
        
        # possessive pronoun generator for beginning of sentences
        def its():
            return random.choice([
                'Its',
                'The tank\'s',
                f'The {tank_name}\'s',
            ])

        # possible openings for the first sentence of the description
        initial = [
            f' The {tank_name} tank is',
            f' The {tank_name} armored vehicle is',
            f' The {tank_name} tank was',
            f' The {tank_name} armored vehicle was'
        ]

        # possible first words of a description sentence
        # duplicates are intentional to increase their odds of being selected
        sentence_starters = [
            f' {it()} has',
            f' {it()} has',
            f' {it()} lacks',
            f' {it()} was',
            f' {it()} was',
            f' {it()} was developed',
            f' {it()} is known',
            f' {it()} is considered',
            f' {it()} is often',
            f' {it()}',
            f' {its()} appearance',
            f' {its()} maneuverability',
            f' {its()} size',
            f' {its()} speed',
            f' {its()} ability to',
            f' {its()}',
            ' Notably,',
            ' Historically,',
            ' The armaments',
            ' The engine',
            ' The armor',
            ' This design',
            # these blanks will let the generator start a sentence however it wants
            '', '', '', '', ''
        ]

        # choose 5-7 sentence openings and mix them up
        sentence_starters = random.Random().sample(sentence_starters, random.randint(5, 7))
        # prepend the first sentence opening
        sentence_starters = [random.choice(initial)] + sentence_starters

        # these are the requirements that are applied to each sentence
        def constraints(text):
            if text == '': return False
            if not text[0].isupper(): return False
            for letter in ['"', '{', '_', '[']:
                if letter in text: return False
            for game_term in ['game', 'player']:
                if game_term in text: return False
            # sometimes it generates long lines of dashes
            if '---' in text: return False
            if not text.endswith('.'): return False
            return True

        # this preamble is included at the beginning of the text to help the
        # generator along, but is not included in the description output
        preamble = 'This is a description of the armored tank {}:'
        output = ''
        
        for starter in sentence_starters:
            if output == '':
                # sentence() can return a blank string if it runs out of attempts at passing the constraints
                # for the first sentence, that is not ok, so keep rolling till we get something
                while output == '':
                    output = self.sentence(f'{preamble} {output}', starter, constraints=constraints)
            else:
                next_sentence = self.sentence(f'{preamble} {output}', starter, constraints=constraints)
                output = ' '.join([output, next_sentence])

        return output