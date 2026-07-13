# week 3 friday
import geopandas as geopd
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon, Point, LineString, MultiPoint, GeometryCollection
from shapely.plotting import plot_polygon, plot_points
import geopandas as gpd

geodata = geopd.read_file('steminist/sc_congressional_district.geojson')
geocensus_data = geopd.read_file('steminist/sc_pl2020_p1.geojson')
geocensus_data = geocensus_data.to_crs(geodata.crs)
  
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

def calculate_pop_center(dist_polygon, census_gdf):
    census_points = {}
    census_points_min = {}
    tot_pop = 1
    tot_min_pop = 1

    for row in census_gdf.itertuples():
        polygon = row.geometry

        if polygon.within(dist_polygon):
            pop = row.P0010001
            min_pop = row.P0010002
            tot_pop += pop
            tot_min_pop += min_pop

            census_points[row.GEOID] = {
                "population": pop,
                "x": polygon.centroid.x,
                "y": polygon.centroid.y
            }
            census_points_min[row.GEOID] = {
                "population": min_pop,
                "x": polygon.centroid.x,
                "y": polygon.centroid.y
            }

    sumx = sum(v["population"] * v["x"] for v in census_points.values())
    sumy = sum(v["population"] * v["y"] for v in census_points.values())
    
    sum_minx = sum(v["population"] * v["x"] for v in census_points_min.values())
    sum_miny = sum(v["population"] * v["y"] for v in census_points_min.values())

    x_coor = sumx/tot_pop
    y_coor = sumy/tot_pop

    x_mincoor = sum_minx/tot_min_pop
    y_mincoor = sum_miny/tot_min_pop

    return [(x_mincoor, y_mincoor),(x_coor, y_coor)]

def pop_center_outside_district(pop_center, dist_polygon):
    new_polygon_coords = []
    polygon_exterior = list(dist_polygon.exterior.coords)

    for vertex in polygon_exterior:
        line = LineString([pop_center, vertex])
        if not multiple_intersection(line, dist_polygon):
            new_polygon_coords.append(vertex)
    new_polygon = Polygon(new_polygon_coords)
    return new_polygon

def geo_copr_polygon(dist_polygon, geo_center):
    new_geopoly_coords = []
    polygon_ext = list(dist_polygon.exterior.coords)
    tol = 1e-6


    for vertex in polygon_ext:
        geoline = LineString([geo_center, vertex])
        if not multiple_intersection(geoline, dist_polygon):
            new_geopoly_coords.append(vertex)
            new_geopt = extend_ray_to_boundary(geo_center, vertex, dist_polygon)
            if new_geopt is not None:
                extension = LineString([vertex, (new_geopt.x, new_geopt.y)])
                if dist_polygon.covers(extension):
                    new_geopoly_coords.append((new_geopt.x, new_geopt.y))
    new_geo_polygon = Polygon(new_geopoly_coords)

    return new_geo_polygon


districts = 0

district_data = {}
fig, ax = plt.subplots()

colors = ['black','pink','orange','cyan','red','brown','olive','yellow','purple']
while districts < 7:
    
    #get district bounds and make polygon
    dist_polygon = geodata.geometry.iloc[districts]
    
    #find total area of the district and convert to miles
    total_area_sq_miles = dist_polygon.area / 2589988.110336

    #find pop center by calling function
    centers = calculate_pop_center(dist_polygon, geocensus_data)
    pop_center = Point(centers[0]) 
    tot_pop_center = Point(centers[1])


    if dist_polygon.contains(pop_center):

        #get new polygon based on pop center
        new_polygon_coords = []
        polygon_exterior = list(dist_polygon.exterior.coords)

        for vertex in polygon_exterior:
            line = LineString([pop_center, vertex])

            if not multiple_intersection(line, dist_polygon):
                new_polygon_coords.append(vertex)
                new_pt = extend_ray_to_boundary(pop_center, vertex, dist_polygon)
                if new_pt is not None:
                    extension = LineString([vertex, (new_pt.x, new_pt.y)])
                    if dist_polygon.covers(extension):
                        new_polygon_coords.append((new_pt.x, new_pt.y))
        new_polygon = Polygon(new_polygon_coords)
        good_area = new_polygon.area / 2589988.110336
    else:
        new_polygon = pop_center_outside_district(pop_center, dist_polygon)
        good_area = new_polygon.area / 2589988.110336
    
    COPr_vals = []
    COPr = good_area/total_area_sq_miles

    # geo_center = dist_polygon.centroid
    # new_geo_polygon = geo_copr_polygon(dist_polygon, geo_center)
    # good_geo_area = new_geo_polygon.area / 2589988.110336
    # geo_COPr = good_geo_area/total_area_sq_miles

    # difference = abs(geo_COPr - COPr)
    # final_COPr = (COPr - difference)
    print("Black Pop COPr: ",COPr)
    # print("Geometric Center different COPr: ",final_COPr)

    COPr_vals.append(COPr)

    districts += 1

    # x, y = dist_polygon.exterior.xy
    # xg, yg = new_geo_polygon.exterior.xy
    # xp, yp = new_polygon.exterior.xy
    # plt.plot(x, y, color='blue', linewidth=2)
    # plt.plot(xg, yg, color='green', linewidth=2)
    # plt.plot(xp, yp, color=colors[districts], linewidth=2)

    plot_polygon(dist_polygon,ax=ax, color="grey", edgecolor="black",linewidth=0.5)
    dx, dy = zip(*dist_polygon.exterior.coords)
    ax.scatter(dx, dy, s=1, color='black')
    #plot_polygon(new_geo_polygon, ax=ax, color='blue', edgecolor='black')
    
    plot_polygon(new_polygon, ax=ax, color= colors[districts], edgecolor='black',linewidth=0.5)
    xs, ys = zip(*new_polygon.exterior.coords)
    ax.scatter(xs, ys, s=1, color='black') 

    plot_points(pop_center, color="black", marker='o',markersize=3)
    plot_points(tot_pop_center, color="black", marker='o',markersize = 3)
    ax.text(
        pop_center.x, pop_center.y,
        f"District = {districts:.3f}\nBlack Pop COPr = {COPr:.3f}",
            fontsize=6,
            ha='right',
            va='top',
            color='black',
            bbox=dict(facecolor='white', alpha=0.1, edgecolor='none')
    )
 
plt.show()
# if the value is close to 1, the shape is compact
# if the value is close to 0, the shape is not compact
'''''
    Geometric score is mostly likely going to be the greater than the pop score
to help the score be a better representation of the shape of the district we find the 
difference of the 2. This shows the difference from what the 'best' possible outcome 
to what was actually done with teh popcenter/shape of district. This will be subtracted 
from the overall score to help penalize the score and show more variance in scores 
from district to district.
    For popcenters that are outside of the district, we will go negative instead of 0-1. 
with a negative score we can obviously assume that the district is not passing the 
compactness score.

'''''