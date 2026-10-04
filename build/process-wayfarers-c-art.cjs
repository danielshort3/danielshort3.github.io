'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'asset-sources/wayfarers-guild/c-production');
const target = path.join(root, 'img/wayfarers-guild');
const areas = require('../js/games/wayfarers-guild/progression-content.js').AREAS;
const skills = require('../js/games/wayfarers-guild/area-skills-content.js').SKILLS;
const hash = buffer => crypto.createHash('sha256').update(buffer).digest('hex');
async function main() {
  const entries = [];
  for (const area of areas) {
    const input = path.join(source, area.id + '.png');
    const meta = await sharp(input).metadata();
    const ids = area.tracks.map(track => 'track-' + area.id + '-' + track.id)
      .concat(skills.filter(skill => skill.areaId === area.id).map(skill => 'skill-' + skill.id));
    for (let index = 0; index < ids.length; index += 1) {
      const col = index % 3, row = Math.floor(index / 3);
      const left = Math.round(col * meta.width / 3), top = Math.round(row * meta.height / 5);
      const width = Math.round((col + 1) * meta.width / 3) - left;
      const height = Math.round((row + 1) * meta.height / 5) - top;
      const file = 'c-' + ids[index] + '.webp';
      const bytes = await sharp(input).extract({left, top, width, height})
        .resize(128, 128, { fit: 'contain', kernel: 'nearest', background: {r:0,g:0,b:0,alpha:0} })
        .webp({lossless:true}).toBuffer();
      fs.writeFileSync(path.join(target, file), bytes);
      entries.push({id:ids[index], file, source:area.id+'.png', cell:index, sha256:hash(bytes)});
    }
  }
  const scenePath = path.join(source, 'scenes.png');
  const meta = await sharp(scenePath).metadata();
  for (let index = 0; index < areas.length; index += 1) {
    const col = index % 2, row = Math.floor(index / 2);
    // Exclude the thin separators produced by the source atlas.
    const left = Math.round(col * meta.width / 2) + 2, top = Math.round(row * meta.height / 3) + 2;
    const width = Math.round((col + 1) * meta.width / 2) - left - 2;
    const height = Math.round((row + 1) * meta.height / 3) - top - 2;
    const file = 'c-scene-' + areas[index].id + '.webp';
    const bytes = await sharp(scenePath).extract({left,top,width,height}).webp({lossless:true}).toBuffer();
    fs.writeFileSync(path.join(target,file),bytes);
    entries.push({id:'scene-'+areas[index].id,file,source:'scenes.png',cell:index,sha256:hash(bytes)});
  }
  fs.writeFileSync(path.join(target,'c-art.json'),JSON.stringify({version:1,tool:'built-in image_gen',direction:'C Living production chain',entries},null,2)+'\n');
  process.stdout.write('Exported '+entries.length+' C assets.\n');
}
main().catch(error => { process.stderr.write(error.stack+'\n'); process.exitCode=1; });
