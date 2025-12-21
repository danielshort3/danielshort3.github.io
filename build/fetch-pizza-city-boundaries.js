const fs = require('fs');
const path = require('path');

const OUTPUT_PATH = path.join(__dirname, '..', 'aws', 'pizza-tips-predict', 'city-boundaries.json');
const CITIES = [
  'Frisco',
  'Plano',
  'The Colony',
  'Lewisville',
  'Carrollton',
  'McKinney',
  'Allen'
];

const USER_AGENT = 'pizza-tips-demo/1.0 (contact: danielshort.me)';
const BASE_URL = 'https://nominatim.openstreetmap.org/search';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function polygonArea(coords) {
  let area = 0;
  for (let i = 0; i < coords.length - 1; i++) {
    const [x1, y1] = coords[i];
    const [x2, y2] = coords[i + 1];
    area += (x1 * y2) - (x2 * y1);
  }
  return Math.abs(area) / 2;
}

function featureArea(feature) {
  if (!feature || !feature.geometry) return 0;
  const { type, coordinates } = feature.geometry;
  if (type === 'Polygon') {
    if (!coordinates.length) return 0;
    return polygonArea(coordinates[0]);
  }
  if (type === 'MultiPolygon') {
    return coordinates.reduce((sum, polygon) => {
      if (!polygon.length) return sum;
      return sum + polygonArea(polygon[0]);
    }, 0);
  }
  return 0;
}

async function fetchCity(city) {
  const params = new URLSearchParams({
    city,
    state: 'Texas',
    country: 'USA',
    format: 'geojson',
    polygon_geojson: '1',
    addressdetails: '1'
  });
  const url = `${BASE_URL}?${params.toString()}`;
  const res = await fetch(url, { headers: { 'User-Agent': USER_AGENT } });
  if (!res.ok) throw new Error(`Failed ${city}: ${res.status}`);
  const data = await res.json();
  const features = (data.features || [])
    .filter(f => f.geometry && (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon'))
    .map(f => ({ ...f, _area: featureArea(f) }));

  if (!features.length) throw new Error(`No polygon features for ${city}`);

  features.sort((a, b) => b._area - a._area);
  const best = features[0];
  const props = best.properties || {};
  return {
    type: 'Feature',
    properties: {
      city,
      display_name: props.display_name || city,
      source: 'OpenStreetMap (Nominatim)'
    },
    geometry: best.geometry
  };
}

async function main() {
  const features = [];
  for (const city of CITIES) {
    const feature = await fetchCity(city);
    features.push(feature);
    await sleep(1100);
  }
  const collection = { type: 'FeatureCollection', features };
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(collection, null, 2));
  console.log(`Saved ${features.length} city boundaries to ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
