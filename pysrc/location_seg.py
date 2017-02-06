"""
build up kd tree for predefined locations
assigning geo-tagged tweets to predefined locations based on nearest neighbor
"""

import kdtree
import json
import os
import pprint

location_dict = dict()
location_tree = None


# build up dictionary for searching locations by "lat" and "lon": a two levels tree {lat: {lon: [location entity]}}
with open(os.path.join(os.pardir, 'data', 'chicago_locations.json')) as fp:
	locations = json.load(fp)
for item in locations:
	tmp = location_dict.setdefault(item["lat"], {})
	tmp[item["lon"]] = item

# build up kdtree for searching nearest location
location_tree = kdtree.create(dimensions=2)
for item in locations:
	location_tree.add((item["lat"], item["lon"]))
if not location_tree.is_balanced:
	location_tree.rebalance()


# loading tweets and assign to nearest location. Result stored as {"name":str, "lon":float, "lat":float, "tweets":list}
with open(os.path.join(os.pardir, "data", "chicago75000s.txt")) as fp:
	for line in fp:
		tweet = json.loads(line)
		nn = location_tree.search_nn((tweet["coordinates"]["coordinates"][1], tweet["coordinates"]["coordinates"][0]))
		# nn will be a tuple like ((<KDNode - (41.854081, -87.6078405)>, distance))
		nn_location = location_dict[nn[0].data[0]][nn[0].data[1]]
		nn_location.setdefault("text", []).append(tweet["text"])
pprint.pprint(location_dict)

# save to file
with open(os.path.join(os.pardir, "data", "chicago75000s_min_assigned.json"), "w") as fp:
	for lat in location_dict:
		for lon in location_dict[lat]:
			json.dump(location_dict[lat][lon], fp)
			fp.write('\n')

