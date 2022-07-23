import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
import json

parser = argparse.ArgumentParser()
parser.add_argument('longitude', metavar='LON', type=float, help='Longitude, deg')
parser.add_argument('latitude',  metavar='LAT', type=float, help='Latitude, deg')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.longitude, args.latitude)
    
with nc.netcdf_file('MSR-2.nc', mmap=False) as netcdf_f:
    variables = netcdf_f.variables

lon = args.longitude
lat = args.latitude

index_lon = np.searchsorted(variables['longitude'].data, lon)
index_lat = np.searchsorted(variables['latitude'].data, lat)

z = variables['Average_O3_column'].data[:, index_lat , index_lon]

max_jan = np.max(variables['Average_O3_column'].data[np.arange(0,468, 12), index_lat , index_lon])
min_jan = np.min(variables['Average_O3_column'].data[np.arange(0,468, 12), index_lat , index_lon])
max_jun = np.max(variables['Average_O3_column'].data[np.arange(6,468, 12), index_lat , index_lon])
min_jun = np.min(variables['Average_O3_column'].data[np.arange(6,468, 12), index_lat , index_lon])
sum_jan = np.sum(variables['Average_O3_column'].data[np.arange(0,468, 12), index_lat , index_lon])
sum_jun = np.sum(variables['Average_O3_column'].data[np.arange(6,468, 12), index_lat , index_lon])


fig = plt.gcf()
plt.grid()
fig.set_size_inches(15,5)

plt.plot(np.arange(0,468, 12), variables['Average_O3_column'].data[np.arange(0,468, 12), index_lat , index_lon], label="jan")
plt.plot(np.arange(0,468, 12), variables['Average_O3_column'].data[np.arange(6,468, 12), index_lat , index_lon], color = "red", label="jun")
plt.plot(variables['time'].data - 108, z,color = "grey", label="all")
plt.scatter(np.arange(0,468, 12), variables['Average_O3_column'].data[np.arange(0,468, 12), index_lat , index_lon])
plt.scatter(np.arange(0,468, 12), variables['Average_O3_column'].data[np.arange(6,468, 12), index_lat , index_lon], color = "red")
plt.legend(loc='best')
plt.title('содержание озона с 1979 до 2019')
plt.xlabel('месяцы, считая от 1979 года')
plt.ylabel('Допсон')

fig.savefig('ozon.png')



file = { 'coordinates': [lon, lat], 
        'jan': {'min': float('{:.1f}'.format(min_jan)),
                'max': float('{:.1f}'.format(max_jan)),
                'mean': float('{:.1f}'.format(sum_jan/39))},
        'jul': {
            'min': float('{:.1f}'.format(min_jun)),
            'max': float('{:.1f}'.format(max_jun)),
            'mean': float('{:.1f}'.format(sum_jun/39))},
        'all': {
            'min': float('{:.1f}'.format(min(z))),
            'max': float('{:.1f}'.format(max(z))),
            'mean': float('{:.1f}'.format(np.mean(z)))}}

with open('ozon.json', 'w') as f:
    json.dump(file, f)
