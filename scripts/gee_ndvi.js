/**
 * GEE Script: NDVI for Karachi
 * Dataset: MODIS MOD13A1.061 — 16-day 500m NDVI composites
 * Export: 16-day mean NDVI for Karachi bounding box → Google Drive CSV
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

var modisNDVI = ee.ImageCollection('MODIS/061/MOD13A1')
  .filterDate(startDate, endDate)
  .filterBounds(karachi)
  .select(['NDVI', 'SummaryQA']);

// Scale factor: 0.0001; filter to good/marginal quality pixels
function scaleNDVI(image) {
  var qa = image.select('SummaryQA');
  var goodQuality = qa.lte(1); // 0=good, 1=marginal
  var ndvi = image.select('NDVI')
    .multiply(0.0001)
    .updateMask(goodQuality)
    .rename('NDVI');
  return ndvi.set('system:time_start', image.get('system:time_start'));
}

var ndviScaled = modisNDVI.map(scaleNDVI);

// Reduce each composite to mean over Karachi bbox
var ndviTimeSeries = ndviScaled.map(function(image) {
  var date  = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: karachi,
    scale: 500,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    date: date,
    NDVI: stats.get('NDVI')
  });
});

var filtered = ndviTimeSeries.filter(ee.Filter.notNull(['NDVI']));

Export.table.toDrive({
  collection:   filtered,
  description:  'karachi_ndvi_2018_2024',
  fileFormat:   'CSV',
  selectors:    ['date', 'NDVI']
});

print('NDVI time series (first 10):', filtered.limit(10));

// Optional: visualise on map
Map.centerObject(karachi, 10);
var ndviVis = {min: -0.1, max: 0.5, palette: ['brown', 'white', 'green']};
Map.addLayer(ndviScaled.mean(), ndviVis, 'Mean NDVI');
Map.addLayer(karachi, {color: 'white'}, 'Karachi Bbox');
