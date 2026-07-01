# week 3 friday
import geopandas as geopd
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, Point, LineString, MultiPoint, GeometryCollection
from shapely.plotting import plot_polygon, plot_points

geodata = geopd.read_file('MN_precincts.geojson')
geojsondata = json.load(open('MN_precincts.geojson'))
def multiple_intersection(line, polygon):
    intersection = line.intersection(polygon.exterior)
    if isinstance(intersection, MultiPoint):
        return len(intersection.geoms) > 1
    elif isinstance(intersection, GeometryCollection):
        return len(intersection.geoms) > 1
    elif isinstance(intersection, LineString):
        return False
    return False

def extend_ray_to_boundary(center, target, polygon, factor=1000):
    cx, cy = center.x, center.y
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    # extend far beyond the polygon
    end = (cx + factor * dx, cy + factor * dy)
    ray = LineString([(cx, cy), end])
    inter = ray.intersection(polygon.boundary)
    pts = []
    if isinstance(inter, Point):
        pts = [inter]
    elif isinstance(inter, MultiPoint):
        pts = list(inter.geoms)
    elif isinstance(inter, GeometryCollection):
        pts = [g for g in inter.geoms if isinstance(g, Point)]
    if not pts:
        return None
    # return the farthest intersection from the center
    return max(pts, key=lambda p: center.distance(p))

precinct_lists = geojsondata['features'][3290]['geometry']['coordinates']
precinct_coords = []
for i in precinct_lists: 
    for j in i: 
        coord = tuple(j)
        precinct_coords.append(coord)
polygon = Polygon(precinct_coords)
total_area_sq_miles = polygon.area / 2589988.110336

districts = geodata[['CONGDIST', 'geometry']]
print(districts)


# find population center
    # compile census blocks
    # find the weight of each census block
    # take the weighted average of the block weights = pop_center
        # 1/ sum of weights * sum(weight* block coords (x)) = pop_center_x
        # 1/ sum of weights * sum(weight* block coords (y)) = pop_center_y
    # pop_center = (pop_center_x, pop_center_y)

pop_center = Point(569194, 5185747)
#pop_center = Point(567431, 5186793)

new_polygon_coords = []
polygon_exterior = list(polygon.exterior.coords)
tol = 1e-6

for vertex in polygon_exterior:
    line = LineString([pop_center, vertex])
    if not multiple_intersection(line, polygon):
        new_polygon_coords.append(vertex)
        new_pt = extend_ray_to_boundary(pop_center, vertex, polygon)
        if new_pt is not None:
            extension = LineString([vertex, (new_pt.x, new_pt.y)])
            if polygon.covers(extension):
                new_polygon_coords.append((new_pt.x, new_pt.y))
new_polygon = Polygon(new_polygon_coords)
good_area = new_polygon.area / 2589988.110336

COPr = good_area/total_area_sq_miles
print(COPr)
fig, ax = plt.subplots()

plot_polygon(polygon,ax=ax, color="blue", edgecolor="black")
plot_polygon(new_polygon, ax=ax, color='purple', edgecolor='black')
plot_points(pop_center, color="black", marker='o')
plt.show()
    # if the value is close to 1, the shape is compact  
    # if the value is close to 0, the shape is not compact