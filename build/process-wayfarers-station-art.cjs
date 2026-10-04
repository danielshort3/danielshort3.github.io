'use strict';

// Only extract, trim and nearest-neighbor resize generated masters. No drawn stand-ins.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const content = require('../js/games/wayfarers-guild/station-content.js');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'asset-sources/wayfarers-guild/stations');
const output = path.join(root, 'img/wayfarers-guild');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const roles = ['miner','porter','scout','artisan','scholar','sailor'];

async function processArt() {
  const config = JSON.parse(fs.readFileSync(path.join(source, 'sources.json')));
  const manifest = {version:1, logicalWidth:384, stationHeight:208, surfaceHeight:320, areas:{}, icons:{}, workers:{file:'station-workers.webp',cellWidth:64,cellHeight:96,columns:4,roles}, records:[]};
  async function emit(master, rect, name, width, height, trim = false) {
    const bytes = fs.readFileSync(path.join(source, master));
    let pipeline = sharp(await sharp(bytes).extract(rect).png().toBuffer());
    if (trim) pipeline = pipeline.trim({threshold:18});
    const data = await pipeline.resize(width,height,{fit:trim?'contain':'fill',kernel:'nearest',background:'#00000000'}).webp({lossless:true,effort:6}).toBuffer();
    fs.writeFileSync(path.join(output,name),data);
    manifest.records.push({file:name,source:master,sourceSha256:sha(bytes),crop:rect,width,height,sha256:sha(data),bytes:data.length});
    return name;
  }
  for (const area of content.AREAS) {
    const spec = config.areas[area.id];
    const metadata = await sharp(path.join(source,spec.file)).metadata();
    const columns = spec.columns || [0,1/3,2/3,1];
    const rows = spec.rows || [0,1/6,2/6,3/6,4/6,5/6,1];
    function cell(col,row) {
      const left = Math.round(columns[col]*metadata.width)+2, top = Math.round(rows[row]*metadata.height)+2;
      const strip = area.id === 'greenway' && col === 0 && row === 1 ? 6 : 0;
      return {left,top:top+strip,width:Math.round(columns[col+1]*metadata.width)-left-2,height:Math.round(rows[row+1]*metadata.height)-top-2-strip};
    }
    const skyMaster = config.skies?.file || spec.file;
    const skyMetadata = await sharp(path.join(source,skyMaster)).metadata();
    const skyIndex = config.skies?.areaOrder.indexOf(area.id) ?? 0;
    const skyRect = config.skies ? {left:2,top:Math.round(skyIndex*skyMetadata.height/6)+2,width:skyMetadata.width-4,height:Math.round((skyIndex+1)*skyMetadata.height/6)-Math.round(skyIndex*skyMetadata.height/6)-4} : cell(0,0);
    const kit = {sky:await emit(skyMaster,skyRect,`station-${area.id}-sky.webp`,384,112),underground:await emit(spec.file,cell(1,0),`station-${area.id}-depth.webp`,384,208),stations:{}};
    for (const station of content.STATIONS.filter(s=>s.areaId===area.id)) {
      const base = `station-${area.id}-${station.localId}`;
      kit.stations[station.id] = {
        background:await emit(spec.file,cell(0,station.index+1),base+'-background.webp',384,208),
        machine:await emit(spec.file,cell(1,station.index+1),base+'-machine.webp',180,112,true),
        addition:await emit(spec.file,cell(2,station.index+1),base+'-addition.webp',96,86,true),
        workerRole:area.id==='greenway'?['scout','porter','scout','porter','porter'][station.index]:area.id==='quarry'?['miner','porter','miner','artisan','scholar'][station.index]:area.id==='workshop'?'artisan':area.id==='harbor'?'sailor':'scholar',
        workerSize:[56,84], anchors:{machine:[256,172],worker:[132,156],addition:[322,172],effect:[149,120]}, width:384,height:208
      };
    }
    manifest.areas[area.id] = kit;
    const iconSpec = config.icons[area.id];
    const iconMetadata = await sharp(path.join(source,iconSpec.file)).metadata();
    const areaSkills = content.SKILLS.filter(s=>s.areaId===area.id);
    const skills = iconSpec.names ? iconSpec.names.map(name => areaSkills.find(skill => skill.name === name)) : areaSkills;
    if (skills.length !== 30 || skills.some(skill => !skill)) throw new Error(`Icon subject map incomplete for ${area.id}`);
    for (let index=0;index<skills.length;index+=1) {
      const x=index%5,y=Math.floor(index/5), cw=iconMetadata.width/5,ch=iconMetadata.height/6;
      const crop={left:Math.round(x*cw)+3,top:Math.round(y*ch)+3,width:Math.round((x+1)*cw)-Math.round(x*cw)-6,height:Math.round((y+1)*ch)-Math.round(y*ch)-6};
      manifest.icons[skills[index].icon]=await emit(iconSpec.file,crop,skills[index].icon+'.webp',128,128,true);
    }
  }
  const cartMaster = config.areas.quarry.file;
  const cart = await emit(cartMaster,{left:420,top:436,width:414,height:194},'station-cart.webp',82,54,true);
  manifest.movingCart = {file:cart,width:82,height:54};
  const workerSpec=config.workers;
  const workerMetadata=await sharp(path.join(source,workerSpec.file)).metadata();
  const composites=[];
  for(let row=0;row<6;row+=1)for(let col=0;col<4;col+=1) {
    const cw=workerMetadata.width/4,ch=workerMetadata.height/6;
    const crop={left:Math.round(col*cw),top:Math.round(row*ch),width:Math.floor(cw),height:Math.floor(ch)};
    const data=await sharp(path.join(source,workerSpec.file)).extract(crop).resize(64,96,{fit:'fill',kernel:'nearest',background:'#00000000'}).png().toBuffer();
    composites.push({input:data,left:col*64,top:row*96});
  }
  const workers=await sharp({create:{width:256,height:576,channels:4,background:'#00000000'}}).composite(composites).webp({lossless:true}).toBuffer();
  fs.writeFileSync(path.join(output,'station-workers.webp'),workers);
  manifest.records.push({file:'station-workers.webp',source:workerSpec.file,sourceSha256:sha(fs.readFileSync(path.join(source,workerSpec.file))),width:256,height:576,sha256:sha(workers),bytes:workers.length});
  fs.writeFileSync(path.join(output,'station-art.json'),JSON.stringify(manifest,null,2)+'\n');
  const runtime={...manifest}; delete runtime.records;
  fs.writeFileSync(path.join(root,'js/games/wayfarers-guild/station-art.js'),`(function(root){'use strict';const art=${JSON.stringify(runtime)};if(typeof module==='object'&&module.exports)module.exports=art;root.WayfarersStationArt=art;})(typeof globalThis!=='undefined'?globalThis:this);\n`);
  console.log(`Exported ${manifest.records.length} layered station assets and ${Object.keys(manifest.icons).length} stable icon mappings.`);
}
if(require.main===module)processArt().catch(error=>{console.error(error);process.exitCode=1;});
module.exports={processArt};
