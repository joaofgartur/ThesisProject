from datasets import GermanCredit, AdultIncome
from metrics import disparate_impact

if __name__ == '__main__':
    german_credit = GermanCredit()
    print(disparate_impact(german_credit))
    adult_income = AdultIncome()
    print(disparate_impact(adult_income))
