import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


#load shape file with geopandas
shapefile = gpd.read_file("FL/FL.shp")

numDistricts = int(max(shapefile["cd113"]))

#extract centroids from shape file
centers = []
weights = []
for index, line in shapefile.iterrows():
	geo = line['geometry']
	pop = line['TotPop']
	centers.append([geo.centroid.x, geo.centroid.y])
	weights.append(pop)

#simulate a certain number of elections
electionRecords = dict()
numSims = 1000
for i in range(numSims):
	#run k-means clustering
	kmeans = KMeans(n_clusters=numDistricts, n_init=1, random_state=None, max_iter=100)
	weighted_kmeans = kmeans.fit(centers, sample_weight = weights)
	group_indicies = kmeans.predict(centers)
	cluster_centers = weighted_kmeans.cluster_centers_

	shapefile['sim district'] = group_indicies  #note: wrapping in tuple makes colors better

	np_shapefile = shapefile.to_numpy()

	demReps = 0
	repReps = 0
	#count up the number of democratic and republican votes in each district
	for districtNum in range(numDistricts):
		district = np_shapefile[np.where(np_shapefile[:, 12] == districtNum)]
		republicanVotes = np.sum(district[:, 9])
		democraticVotes = np.sum(district[:, 10])
		if republicanVotes < democraticVotes:
			demReps += 1
		else:
			repReps += 1

	#record this election
	if demReps not in electionRecords:
		electionRecords[demReps] = 0

	electionRecords[demReps] += 1

	print(f"iteration: {i}: {demReps} democrats and {repReps} republicans elected")

print(electionRecords)
plt.bar(electionRecords.keys(), electionRecords.values(), color ='darkblue', width = 0.8)

plt.xlabel("No. of Democratic Represenatives Elected")
plt.ylabel(f"No. of simulations (out of {numSims})")

#plot the simulated districts
f, ax = plt.subplots(1)
shapefile.plot(ax=ax, column='sim district')

# shapefile.boundary.plot(ax=ax, color=(0,0,0)) #plot boundaries (makes graph messy)

#plot centroids of census districts
# ax.scatter([c[0] for c in centers], [c[1] for c in centers], marker="o", color='r')

#plot means from k-means clustering
ax.scatter([c[0] for c in cluster_centers], [c[1] for c in cluster_centers], marker="o", color='black')

plt.show()
