import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import searborn as sns
from sklearn.decomposition import PCA

def showPCA2D(X, Y):
	"""

	"""
	if type(X) is pd.DataFrame:
		X = X.values
	if type(Y) is pd.Series:
		Y = Y.values

	X_pca = PCA().fit_transform(X)
	columns = [f"PC1 ({round(X_pca.explained_variance_ratio_[0],2)*100}%)",
			   f"PC2 ({round(X_pca.explained_variance_ratio_[1],2)*100}%)", "Output"]

	df = pd.DataFrame( np.concatenate([X_pca[:,:2], Y.reshape((-1,1))],axis=1) , columns=columns)

	sns.relplot(x=columns[0], y=columns[1], hue="Output", data=df)
	plt.show()