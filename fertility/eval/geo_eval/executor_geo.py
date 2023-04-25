"""An executor for GeoQuery FunQL programs."""
import os.path
from typing import List

from overrides import overrides
from pyswip import Prolog
import re

# from utils.executor import Executor


# @Executor.register("geo_executor")
# @Executor.register("geo_executor")
class ProgramExecutorGeo:

    def __init__(self):
        self._prolog = Prolog()
        fname = __file__
        self._prolog.consult(os.path.join(os.path.dirname(fname), 'geobase.pl'))
        self._prolog.consult(os.path.join(os.path.dirname(fname), 'geoquery.pl'))
        self._prolog.consult(os.path.join(os.path.dirname(fname), 'eval.pl'))

    # @overrides
    def execute(self, program: str) -> str:
        # make sure entities with multiple words are parsed correctly
        # program = re.sub("' (\w+) (\w+) '", "'"+r"\1#\2"+"'", program)
        # program = re.sub("' (\w+) (\w+) (\w+) '", "'" + r"\1#\2#\3" + "'", program)
        program = program.replace(' ','')

        try:
            answers = list(self._prolog.query("eval(" + '{}, X'.format(program) + ").", maxresult=1))
        except Exception as e:
            return 'error_parse: {}'.format(e)
        return str([str(answer) for answer in answers[0]['X']])


if __name__ == "__main__":
    pred_program = "answer ( population_1 ( city ( loc_2 ( stateid ( 'minnesota' ) ) ) ) )"
    executor = ProgramExecutorGeo()
    denotation = executor.execute(pred_program)
    print(denotation)
