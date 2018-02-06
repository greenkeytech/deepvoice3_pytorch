# IPython log file

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pa
import numpy as np
data = np.array(data).reshape(108, 3)[5:]
data = """Common voice	Voice Quality	M/F
3	2	F
3	2	M
2	2	M
2	3	F
3	2	F
3	2	F
3	3	F
2	3	M
3	3	F
3	3	F
2	3	F
1	1	M
3	2	F
3	2	F
3	3	F
2	3	M
3	3	M
2	3	F
3	2	M
3	3	M
3	3	M
2	1	F
3	3	M
2	3	F
2	2	M
3	3	M
3	3	F
2	1	M
3	2	F
3	1	M
3	3	F
3	2	M
3	2	M
3	2	M
3	2	F
2	3	F
2	1	M
1	2	F
2	3	F
3	3	F
3	2	F
3	3	F
3	1	F
3	2	M
2	3	M
2	3	M
3	1	M
3	1	M
2	2	M
3	3	F
3	2	F
3	2	M
2	2	M
3	2	F
2	2	M
2	3	F
3	2	F
3	1	M
2	2	M
3	2	M
3	2	M
3	1	F
2	2	M
3	2	F
3	1	F
2	2	F
3	2	F
3	2	M
3	1	F
3	2	F
3	3	F
3	1	M
3	1	F
3	1	M
3	1	F
3	3	F
3	3	F
3	2	F
3	3	F
3	2	M
3	2	F
3	3	F
3	2	F
3	2	M
3	2	F
3	3	F
3	1	F
3	1	M
3	2	F
3	3	F
2	3	F
2	1	M
2	3	F
2	3	F
2	3	F
2	2	F
3	3	F
3	2	F
3	2	M
2	2	M
2	2	F
2	1	M
2	1	F
2	2	F
3	1	M
2	1	M
2	1	M
2	2	M""".split()
data = np.array(data[5:]).reshape(108, 3)
headers = ["Commonness", "Voice quality", "Perceived gender"]
df = pa.DataFrame(data, columns=headers)
g = sns.FacetGrid(df, row=headers[-1], col=headers[0], margin_titles=True)
g.map(plt.hist, headers[1], bins=range(1, 4))
plt.show()
