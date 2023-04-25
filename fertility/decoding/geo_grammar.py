import itertools
from typing import Dict

import numpy as np
from nltk import CFG, Production, Nonterminal

from fertility.decoding.cnf_helper import to_cnf, compute_derivable_lengths
from fertility.decoding.decoding_grammar import DecodingGrammar, DecodingGrammarFast

# The following grammar is essentially that of Guo et al. 2020, "Benchmarking Meaning Representations in Neural Semantic Parsing"
# https://github.com/JasperGuo/Unimer/blob/master/grammars/geo/typed_funql_grammar.py

# I added two rules in order to parse the data (plus rules involving copying):
# River -> "most" River and
# Query -> "answer" Num

GEO_BASE_GRAMMAR = """
Query -> "answer" City  | "answer" Country  | "answer" Num  | "answer" Place  | "answer" State  | "answer" River 
Country -> "countryid" "usa" | "country" "all" | "each" Country  | "exclude" Country Country  | "intersection" Country Country  | "largest" Country  | "smallest" Country  | "loc_1" City  | "loc_1" Place  | "loc_1" River  | "loc_1" State  | "most" Country  | "traverse_1" River 
State -> "stateid" StateName  | "state" State  | "state" "all" | "smallest" State  | "smallest_one" "area_1" State  | "smallest_one" "density_1" State  | "smallest_one" "population_1" State  | "largest" State  | "largest_one" "area_1" State  | "largest_one" "density_1" State  | "largest_one" "population_1" State  | "each" State  | "exclude" State State  | "intersection" State State  | "fewest" State  | "most" State  | "most" Place  | "most" River  | "most" City  | "next_to_1" State  | "next_to_2" State  | "next_to_2" River  | "traverse_1" River  | "loc_1" River  | "capital_2" City  | "loc_1" City  | "high_point_2" Place  | "low_point_2" Place  | "loc_1" Place  | "loc_2" Country 
StateAbbrev -> "dc" | "pa" | "ga" | "me" | "wa" | "tx" | "ma" | "sd" | "az" | "mn" | "mo"
StateName -> "washington" | "kansas" | "pennsylvania" | "new" "york" | "south" "carolina" | "california" | "west" "virginia" | "kentucky" | "vermont" | "hawaii" | "new" "mexico" | "montana" | "illinois" | "georgia" | "louisiana" | "indiana" | "oklahoma" | "utah" | "arkansas" | "michigan" | "alaska" | "alabama" | "missouri" | "wisconsin" | "wyoming" | "maine" | "florida" | "south" "dakota" | "tennessee" | "north" "carolina" | "new" "jersey" | "minnesota" | "arizona" | "new" "hampshire" | "texas" | "colorado" | "mississippi" | "idaho" | "oregon" | "maryland" | "north" "dakota" | "nebraska" | "rhode" "island" | "ohio" | "massachusetts" | "virginia" | "nevada" | "delaware" | "iowa"
City -> "city" "all" | "city" City  | "loc_2" State   | "loc_2" Country  | "capital" City  | "capital" Place  | "capital" "all" | "capital_1" Country  | "capital_1" State  | "cityid" CityName StateAbbrev  | "cityid" CityName "_" | "each" City  | "exclude" City City  | "intersection" City City  | "fewest" City  | "largest" City  | "largest_one" "density_1" City  | "largest_one" "population_1" City  | "largest_one" "density_1" City  | "smallest" City  | "smallest_one" "population_1" City  | "loc_1" Place  | "major" City  | "most" City  | "traverse_1" River 
CityName -> "washington" | "minneapolis" | "sacramento" | "rochester" | "indianapolis" | "portland" | "new" "york" | "erie" | "san" "diego" | "baton" "rouge" | "miami" | "kalamazoo" | "durham" | "salt" "lake" "city" | "des" "moines" | "pittsburgh" | "riverside" | "dover" | "chicago" | "albany" | "tucson" | "austin" | "san" "antonio" | "houston" | "scotts" "valley" | "montgomery" | "springfield" | "boston" | "boulder" | "san" "francisco" | "flint" | "fort" "wayne" | "spokane" | "san" "jose" | "tempe" | "dallas" | "new" "orleans" | "seattle" | "denver" | "salem" | "detroit" | "plano" | "atlanta" | "columbus"
Num -> Digit | "area_1" City  | "area_1" Country  | "area_1" Place  | "area_1" State  | "count" City  | "count" Country  | "count" Place  | "count" River  | "count" State  | "density_1" City  | "density_1" Country  | "density_1" State  | "elevation_1" Place  | "population_1" City  | "population_1" Country  | "population_1" State  | "size" City  | "size" Country  | "size" State  | "smallest" Num  | "sum" Num  | "len" River  
Digit -> "0.0" | "1.0" | "0"
Place -> "loc_2" City  | "loc_2" State  | "loc_2" Country  | "each" Place  | "elevation_2" Num  | "exclude" Place Place  | "intersection" Place Place  | "fewest" Place  | "largest" Place  | "smallest" Place  | "highest" Place  | "lowest" Place  | "high_point_1" State  | "low_point_1" State  | "higher_1" Place  | "higher_2" Place  | "lower_1" Place  | "lower_2" Place  | "lake" Place  | "lake" "all" | "mountain" Place  | "mountain" "all" | "place" Place  | "place" "all" | "placeid" PlaceName  | "major" Place 
PlaceName -> "guadalupe" "peak" | "mount" "whitney" | "mount" "mckinley" | "death" "valley"
River -> "river" River  | "loc_2" State  | "loc_2" Country  | "each" River  | "exclude" River River  | "intersection" River River  | "fewest" River  | "longer" River  | "longest" River  | "major" River  | "most" State  | "river" "all" | "riverid" RiverName  | "shortest" River  | "traverse_2" City  | "traverse_2" Country  | "traverse_2" State 
RiverName -> "chattahoochee" | "north" "platte" | "rio" "grande" | "ohio" | "potomac" | "missouri" | "red" | "colorado" | "mississippi" | "delaware"

Copy -> "@COPY@" | "@COPY@" Copy
PlaceName -> Copy
RiverName -> Copy
CityName -> Copy
StateName -> Copy

River -> "most" River
Query -> "answer" Num
"""


GEO_BASE_GRAMMAR = CFG.fromstring(GEO_BASE_GRAMMAR)
GEO_BASE_GRAMMAR = to_cnf(GEO_BASE_GRAMMAR)


@DecodingGrammar.register("geo")
class GeoDecodingGrammar(DecodingGrammarFast):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(GEO_BASE_GRAMMAR)
        # Precomputed to save time.
        self.derivable_lengths = np.array([False, False] + 1000 * [True])


if __name__ == "__main__":
    from nltk.parse.chart import BottomUpChartParser

    def recognize(parser, s, start_symbol):
        chart = parser.chart_parse(s)
        it = chart.parses(start_symbol)
        try:
            t = next(it)
            return t
        except StopIteration:
            return False

    # print(GEO_BASE_GRAMMAR)

    # print(compute_derivable_lengths(GEO_BASE_GRAMMAR, 500))

    parser = BottomUpChartParser(GEO_BASE_GRAMMAR)
    problems = 0
    with open("data/geo/herzig/geo_herzig_train_no_quotes.tsv") as f:
        for line in f:
            nl, mr = line.strip().split("\t")
            mr = mr.split(" ")
            if not recognize(parser, mr, GEO_BASE_GRAMMAR.start()):
                print("Can't parse:")
                print(nl, mr)
                problems += 1
    print(problems)