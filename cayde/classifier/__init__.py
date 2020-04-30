from .classifier import Classifier
from .ada import AdaBoosterClassifier
from .dtree import DecisionTreeClassifier
from .logistic import LogisticClassifier
from .svm import SVMClassifier

try:
    from .tfc import TensorflowClassifier
except ImportError:
    print("Tensorflow Classifier Unavailable")