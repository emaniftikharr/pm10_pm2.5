/**
 * GEE Script: Land Surface Temperature (LST) for Karachi
 * Dataset: MODIS MOD11A1.061 — Daily 1km LST
 * Export: Daily mean LST (°C) for Karachi bounding box → Google Drive CSV
 *
 * HOW TO USE:
 *   1. Open code.earthengine.google.com
 *   2. Paste this script → Run
 *   3. Go to Tasks tab → Run the export task
 *   4. Download from Google Drive → place in data/raw/
 */

var karachi = ee.Geometry.Rectangle([66.85, 24.74, 67.25, 25.10]);

var startDate = '2018-01-01';
var endDate   = '2024-12-31';

var modisLST = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(startDate, endDate)
  .filterBounds(karachi)
  .select(['LST_Day_1km', 'QC_Day']);

// Convert raw DN to Celsius: scale=0.02, offset=-273.15
function toCelsius(image) {
  var qc = image.select('QC_Day');
  var goodQuality = qc.bitwiseAnd(3).eq(0); // bits 0-1 == 00 (good quality)
  var lst = image.select('LST_Day_1km')
    .multiply(0.02)
    .subtract(273.15)
    .updateMask(goodQuality)
    .rename('LST_C');
  return lst.set('system:time_start', image.get('system:time_start'));
}

var lstC = modisLST.map(toCelsius);

// Reduce each image to mean over Karachi bbox
var lstTimeSeries = lstC.map(function(image) {
  var date  = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: karachi,
    scale: 1000,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    date:  date,
    LST_C: stats.get('LST_C')
  });
});

// Remove entries where LST is null (cloud-masked days)
var filtered = lstTimeSeries.filter(ee.Filter.notNull(['LST_C']));

Export.table.toDrive({
  collection:   filtered,
  description:  'karachi_lst_2018_2024',
  fileFormat:   'CSV',
  selectors:    ['date', 'LST_C']
});

print('LST time series (first 10):', filtered.limit(10));

// Optional: visualise on map
Map.centerObject(karachi, 10);
var lstVis = {min: 20, max: 50, palette: ['blue', 'yellow', 'red']};
Map.addLayer(lstC.mean(), lstVis, 'Mean LST (°C)');
Map.addLayer(karachi, {color: 'white'}, 'Karachi Bbox');
