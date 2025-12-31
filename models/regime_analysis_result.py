from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List,Optional,Dict,Tuple
from pandas import DataFrame


@dataclass_json
@dataclass
class RegimeInterpretation:
    label : str
    description : str
    regime_drivers: List[str]
    diversifiers: List[str]
    regime_dependent_factors: List[str]
    most_useful_for_distinction:  List[str]

@dataclass_json
@dataclass
class RegimeAnalysisResult:
    regime_id : int
    regime_dates: List[int]
    factor_stats: Dict
    covariance_matrix: Optional[List[List[float]]] = None
    correlation_matrix: Optional[List[List[float]]] = None
    interpretation: Optional[RegimeInterpretation] = None

@dataclass_json
@dataclass
class RegimeAnalysisReport:
    N: int
    regime_dates : List[Tuple[int,int]]
    factor_names: List[str]
    regimes: List[RegimeAnalysisResult] = None

    def __post_init__(self):
        if self.regimes is None:
            self.regimes = []

from core.utils import read_file_from_s3

if __name__ == '__main__':
    fn = "/tmp/regime_results.json"
    fn = "/Users/alekseymatiychenko/Documents/python_code/risk_lens/models/data/regime/main_regime_model.json"
    text = read_file_from_s3("risk-lens","regimes/main_regime_model.json")
    report = RegimeAnalysisReport.from_json(text)

    for i in range(5):
        print(f"{i}:  {report.regimes[i].interpretation.label} : {report.regimes[i].interpretation.description}")
