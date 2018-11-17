from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.01, C=100)

x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)

print(digits.data[[-5]])
print(digits.data[[-5]].shape)
print("Prediction ", clf.predict(digits.data[[-11]]))

plt.imshow(digits.images[-11], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

class SVM():
    def __init__(self):
        self.digits = datasets.load_digits()
        self.clf = svm.SVC(gamma=0.01, C=100)
    
    def train(self):
        x,y = self.digits.data[:-1], self.digits.target[:-1]
        self.clf.fit(x,y)

    def guess(self, data):
        return self.clf.predict(data)