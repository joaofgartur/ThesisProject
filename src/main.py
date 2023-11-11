from datasets import GermanCredit, AdultIncome, Compas
from metrics import disparate_impact
from algorithms import massaging

if __name__ == '__main__':
    learning_settings = {"train_size": 0.7,"test_size": 0.3, "seed": 125}

    german_credit = GermanCredit("german", ["Attribute9"])
    # print(disparate_impact(german_credit))

    # adult_income = AdultIncome("adult", ["race"])
    # print(disparate_impact(adult_income))

    # compas = Compas("compas", ["race", "sex"])
    # print(disparate_impact(compas))

    massaging(dataset=german_credit, sensitive_attribute="Attribute9", learning_settings=learning_settings)
