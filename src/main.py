from datasets import GermanCredit, AdultIncome, Compas
from metrics import disparate_impact

if __name__ == '__main__':

    german_credit = GermanCredit("german", ["Attribute9"])
    print(disparate_impact(german_credit))

    adult_income = AdultIncome("adult", ["race"])
    print(disparate_impact(adult_income))

    compas = Compas("compas", ["race", "sex"])
    print(disparate_impact(compas))
