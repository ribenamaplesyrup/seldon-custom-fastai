from AdultClassifier import AdultClassifier

X = {
    "data": {
	"names": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"],
    "list": [44," Private",236746," Masters",14," Divorced"," Exec-managerial"," Not-in-family"," White"," Male",10520,0,45," United-States",">=50k"]
    }
}

result = AdultClassifier().predict(X['data']['list'], X['data']['names'])
print(result)
