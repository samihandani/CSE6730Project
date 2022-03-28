import sys
import geopandas as gpd
from dbfread import DBF
import matplotlib.pyplot as plt


table = DBF('FL/FL.dbf')

for record in table:
	print(record)


shapefile = gpd.read_file("FL/FL.shp")

# print(shapefile)

#extract centroids from shape file
xCenters = []
yCenters = []
for geo in shapefile['geometry']:
	xCenters.append(geo.centroid.x)
	yCenters.append(geo.centroid.y)


#plot voting age pop
f, ax = plt.subplots(1)
shapefile.plot(ax=ax, column='VAP')
shapefile.boundary.plot(ax=ax, color=(0,0,0))

#plot centroids
ax.scatter(xCenters, yCenters, marker="o", color='r')

plt.show()
