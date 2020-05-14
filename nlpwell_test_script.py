import cayde
from cayde.well.nlpwell import NLPWell
from cayde.classifier.dtree import DecisionTreeClassifier
from cayde.classifier.ada import AdaBoosterClassifier
from cayde.classifier.svm import SVMClassifier

well = NLPWell("training_data.csv")
well.load_or_fetch()

well.input_cols = ["head", "body"]
well.text_cols = ["head", "body"]
well.output_col = "Stance"
well.df

print(well.createSvdfFeatures())
